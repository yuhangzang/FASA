import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses import binary_cross_entropy, mask_cross_entropy
from mmdet.models.losses import cross_entropy


@LOSSES.register_module()
class CrossEntropyCounterLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 use_cums=False,
                 num_classes=1203):
        """CrossEntropyCounterLoss.
        """
        super(CrossEntropyCounterLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.num_classes = num_classes
        self.use_cums = use_cums
        if self.use_cums:
            self.open_cums()

    def open_cums(self):
        self.use_cums = True
        self.reduction_old = self.reduction
        self.reduction = 'none'
        self.cum_losses = torch.zeros(self.num_classes + 1).cuda()
        self.cum_labels = torch.zeros(self.num_classes + 1).cuda()

    def close_cums(self):
        self.use_cums = False
        self.reduction = self.reduction_old
        self.cum_losses = torch.zeros(self.num_classes + 1).cuda()
        self.cum_labels = torch.zeros(self.num_classes + 1).cuda()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        if self.use_cums:
            unique_labels = label.unique()
            for u_l in unique_labels:
                inds_ = torch.where(label == u_l)[0]
                self.cum_labels[int(u_l)] += len(inds_)
                self.cum_losses[int(u_l)] += loss_cls[inds_].sum()
            loss_cls = loss_cls.mean()

        return loss_cls
