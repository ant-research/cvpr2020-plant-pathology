# @Author: NickYi1990, yelanlan
# @Date: 2020-06-16 20:43:36
# @Last Modified by:   NickYi1990
# @Last Modified time: 2020-06-14 16:21:14
# Third party libraries

import torch.nn as nn
import torch


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))
