# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
from time import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Third party libraries
import torch
from dataset import generate_transforms, generate_dataloaders
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# User defined libraries
from models import se_resnext50_32x4d
from utils import init_hparams, init_logger, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # 让每次模型初始化一致, 不让只要中间有再次初始化的情况, 结果立马跑偏
        seed_reproducer(self.hparams.seed)

        self.model = se_resnext50_32x4d()
        self.criterion = CrossEntropyLossOneHot()
        self.logger_kun = init_logger("kun_in", hparams.log_dir)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch

        scores = self(images)
        loss = self.criterion(scores, labels)
        # self.logger_kun.info(f"loss : {loss.item()}")
        # ! can only return scalar tensor in training_step
        # must return key -> loss
        # optional return key -> progress_bar optional (MUST ALL BE TENSORS)
        # optional return key -> log optional (MUST ALL BE TENSORS)
        data_load_time = torch.sum(data_load_time)

        return {
            "loss": loss,
            "data_load_time": data_load_time,
            "batch_run_time": torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device),
        }

    def training_epoch_end(self, outputs):
        # outputs is the return of training_step
        train_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack([output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack([output["batch_run_time"] for output in outputs]).sum()

        self.current_epoch += 1
        if self.current_epoch < (self.trainer.max_epochs - 4):
            self.scheduler = warm_restart(self.scheduler, T_mult=2)

        return {"train_loss": train_loss_mean}

    def validation_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)

        # must return key -> val_loss
        return {
            "val_loss": loss,
            "scores": scores,
            "labels": labels,
            "data_load_time": data_load_time,
            "batch_run_time": torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device),
        }

    def validation_epoch_end(self, outputs):
        # compute loss
        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack([output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack([output["batch_run_time"] for output in outputs]).sum()

        # compute roc_auc
        scores_all = torch.cat([output["scores"] for output in outputs]).cpu()
        labels_all = torch.round(torch.cat([output["labels"] for output in outputs]).cpu())
        val_roc_auc = roc_auc_score(labels_all, scores_all)

        # terminal logs
        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"lr : {self.scheduler.get_lr()[0]:.6f} | "
            f"val_loss : {val_loss_mean:.4f} | "
            f"val_roc_auc : {val_roc_auc:.4f} | "
            f"data_load_times : {self.data_load_times:.2f} | "
            f"batch_run_times : {self.batch_run_times:.2f}"
        )
        # f"data_load_times : {self.data_load_times:.2f} | "
        # f"batch_run_times : {self.batch_run_times:.2f}"
        # must return key -> val_loss
        return {"val_loss": val_loss_mean, "val_roc_auc": val_roc_auc}


if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)

    # Init Hyperparameters
    hparams = init_hparams()

    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # Load data
    data, test_data = load_data(logger)

    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    # Do cross validation
    valid_roc_auc_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_dataloader, val_dataloader = generate_dataloaders(hparams, train_data, val_data, transforms)

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_roc_auc",
            save_top_k=6,
            mode="max",
            filepath=os.path.join(hparams.log_dir, f"fold={fold_i}" + "-{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}"),
        )
        early_stop_callback = EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", verbose=True)

        # Instance Model, Trainer and train model
        model = CoolSystem(hparams)
        trainer = pl.Trainer(
            gpus=hparams.gpus,
            min_epochs=70,
            max_epochs=hparams.max_epochs,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            progress_bar_refresh_rate=0,
            precision=hparams.precision,
            num_sanity_val_steps=0,
            profiler=False,
            weights_summary=None,
            use_dp=True,
            gradient_clip_val=hparams.gradient_clip_val,
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        valid_roc_auc_scores.append(round(checkpoint_callback.best, 4))
        logger.info(valid_roc_auc_scores)

        del model
        gc.collect()
        torch.cuda.empty_cache()
