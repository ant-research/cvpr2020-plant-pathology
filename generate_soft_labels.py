import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import torch
from dataset import generate_transforms
from sklearn.model_selection import KFold
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import CoolSystem
from utils import init_hparams, init_logger, seed_reproducer, load_data
from dataset import PlantDataset


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

    early_stop_callback = EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", verbose=True)

    # Instance Model, Trainer and train model
    model = CoolSystem(hparams)
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        min_epochs=70,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=0,
        precision=hparams.precision,
        num_sanity_val_steps=0,
        profiler=False,
        weights_summary=None,
        use_dp=True,
        gradient_clip_val=hparams.gradient_clip_val,
    )

    submission = []
    PATH = [
        "logs_submit/fold=0-epoch=42-val_loss=0.1807-val_roc_auc=0.9931.ckpt",
        "logs_submit/fold=1-epoch=46-val_loss=0.1486-val_roc_auc=0.9946.ckpt",
        "logs_submit/fold=2-epoch=47-val_loss=0.1212-val_roc_auc=0.9952.ckpt",
        "logs_submit/fold=3-epoch=41-val_loss=0.1005-val_roc_auc=0.9884.ckpt",
        "logs_submit/fold=4-epoch=66-val_loss=0.1144-val_roc_auc=0.9913.ckpt",
    ]

    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    train_data_cp = []
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)
        val_data_cp = val_data.copy()
        val_dataset = PlantDataset(
            val_data, transforms=transforms["val_transforms"], soft_labels_filename=hparams.soft_labels_filename
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        submission = []
        model.load_state_dict(torch.load(PATH[fold_i])["state_dict"])
        model.to("cuda")
        model.eval()

        for i in range(1):
            val_preds = []
            labels = []
            with torch.no_grad():
                for image, label, times in tqdm(val_dataloader):
                    val_preds.append(model(image.to("cuda")))
                    labels.append(label)

                labels = torch.cat(labels)
                val_preds = torch.cat(val_preds)
                submission.append(val_preds.cpu().numpy())

        submission_ensembled = 0
        for sub in submission:
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        val_data_cp.iloc[:, 1:] = submission_ensembled
        train_data_cp.append(val_data_cp)
    soft_labels = data[["image_id"]].merge(pd.concat(train_data_cp), how="left", on="image_id")
    soft_labels.to_csv("soft_labels.csv", index=False)

    # ==============================================================================================================
    # Generate Submission file
    # ==============================================================================================================
    test_dataset = PlantDataset(
        test_data, transforms=transforms["train_transforms"], soft_labels_filename=hparams.soft_labels_filename
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=hparams.num_workers, pin_memory=True, drop_last=False,
    )

    submission = []
    for path in PATH:
        model.load_state_dict(torch.load(path)["state_dict"])
        model.to("cuda")
        model.eval()

        for i in range(8):
            test_preds = []
            labels = []
            with torch.no_grad():
                for image, label, times in tqdm(test_dataloader):
                    test_preds.append(model(image.to("cuda")))
                    labels.append(label)

                labels = torch.cat(labels)
                test_preds = torch.cat(test_preds)
                submission.append(test_preds.cpu().numpy())

    submission_ensembled = 0
    for sub in submission:
        submission_ensembled += softmax(sub, axis=1) / len(submission)
    test_data.iloc[:, 1:] = submission_ensembled
    test_data.to_csv("submission.csv", index=False)
