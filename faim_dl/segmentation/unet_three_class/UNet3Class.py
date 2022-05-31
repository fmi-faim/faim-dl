import logging

import numpy as np
import torch
from composer import ComposerModel
from composer.models.loss import Dice
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss, MaskedLoss
from typing import Any, Optional, Tuple
from torch import nn
from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.models.base import ComposerModel
from composer.models.unet.model import UNet as UNetModel
from torchmetrics import MetricCollection

log = logging.getLogger(__name__)

__all__ = ["UNet"]


class UNet(ComposerModel):
    """A U-Net model extending :class:`.ComposerModel`.
    See U-Net: Convolutional Networks for Biomedical Image Segmentation (`Ronneberger et al, 2015`_)
    on the U-Net architecture.
    Args:
        num_classes (int): The number of classes. Needed for classification tasks. Default: ``3``.
    .. _Ronneberger et al, 2015: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels=1, out_channels=3, depth=3) -> None:
        super().__init__()

        self.module = self.build_nnunet(in_channels=in_channels, out_channels=out_channels, depth=depth)

        self.train_loss = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True,
                                     ce_weight=torch.from_numpy(np.array([1, 1, 3], dtype=np.float32)))
        self.train_metric = Dice(num_classes=out_channels - 1)
        self.val_metric = Dice(num_classes=out_channels - 1)

    def loss(self, outputs: Any, batch: BatchPair, *args, **kwargs) -> Tensors:
        _, y = batch
        mask = y[:, 3:]
        y = y[:, :3]
        return self.train_loss(outputs * mask, y * mask)

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def metrics(self, train: bool = False) -> Metrics:
        if train:
            return self.train_metric
        else:
            return self.val_metric

    def forward(self, batch: BatchPair) -> Tensor:
        x, _ = batch
        logits = self.module(x)
        return logits

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        assert self.training is False, "For validation, model must be in eval mode"
        image, target = batch
        mask = target[:, 3:]
        target = target[:, :3]
        pred = self.module(image)
        return pred * mask, torch.argmax(target * mask, 1)

    def build_nnunet(self, in_channels, out_channels, depth) -> torch.nn.Module:
        kernels = [[3, 3]] * (depth + 1)
        strides = [[1, 1]] + [[2, 2]] * depth
        model = UNetModel(in_channels=in_channels,
                          n_class=out_channels,
                          kernels=kernels,
                          strides=strides,
                          dimension=2,
                          residual=True,
                          normalization_layer="batch",
                          negative_slope=0.01)

        return model
