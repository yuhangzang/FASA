import torch
import torch.nn as nn
from sklearn.cluster import AffinityPropagation

from .lvis_instances import LVIS_INSTANCES

from mmcv.runner import force_fp32, get_dist_info

from mmdet.core.utils import reduce_mean
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.utils import get_root_logger


@HEADS.register_module()
class ConvFCFASABBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 fasa_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCFASABBoxHead,
              self).__init__(num_shared_convs, num_shared_fcs, num_cls_convs,
                             num_cls_fcs, num_reg_convs, num_reg_fcs,
                             conv_out_channels, fc_out_channels, conv_cfg,
                             norm_cfg, init_cfg, *args, **kwargs)
        self.feature_mean = nn.Parameter(
            torch.zeros(self.num_classes, self.cls_last_dim),
            requires_grad=False)
        self.feature_std = nn.Parameter(
            torch.zeros(self.num_classes, self.cls_last_dim),
            requires_grad=False)
        self.feature_used = nn.Parameter(
            torch.zeros(self.num_classes), requires_grad=False)
        self.decay_ratio = fasa_cfg.get('decay_ratio', 0.1)
        self.loss_aug_weight = fasa_cfg.get('loss_aug_weight', 0.1)
        self.dynamic_up = fasa_cfg.get('dynamic_up', 1.1)
        self.dynamic_down = fasa_cfg.get('dynamic_down', 0.9)

        self.instance_count_list = torch.zeros(self.num_classes).cuda()

        for k, v in LVIS_INSTANCES.items():
            self.instance_count_list[int(k)] = int(v)
        instance_prob_power = fasa_cfg.get('instance_prob_power', 1)
        instance_prob_scale = fasa_cfg.get('instance_prob_scale', 1)
        instance_prob = 1 / self.instance_count_list
        instance_prob = instance_prob_scale * torch.pow(
            instance_prob / instance_prob.sum(), instance_prob_power)
        instance_prob = instance_prob.clamp(0, 1)
        self.prob_list = nn.Parameter(instance_prob, requires_grad=False)

        # Cum Loss
        self.cum_loss_perclass_t0 = nn.Parameter(
            torch.zeros(self.num_classes + 1), requires_grad=False).cuda()
        self.cum_loss_perclass_t1 = nn.Parameter(
            torch.zeros(self.num_classes + 1), requires_grad=False).cuda()

        self.tf_writer = None
        self.epoch = 0
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            return cls_score, bbox_pred, x_cls
        else:
            return cls_score, bbox_pred

    def fa_update(self, embedding, labels):
        if len(labels) > 0:
            uniq_c = torch.unique(labels)
            for c in uniq_c:
                c = int(c)
                select_index = torch.nonzero(
                    labels == c, as_tuple=False).squeeze(1)

                embedding_temp = embedding[select_index]

                self.fa_update_push(embedding_temp.detach(), c)
        return

    def fa_update_push(self, embedding, labels):
        mean = embedding.mean(dim=0)
        var = embedding.var(dim=0, unbiased=False)
        n = embedding.numel() / embedding.size(1)
        if n > 1:
            var = var * n / (n - 1)
        else:
            var = var
        if self.feature_used[labels] > 0:
            with torch.no_grad():
                self.feature_mean[labels] = self.decay_ratio * mean + (
                    1 - self.decay_ratio) * self.feature_mean[labels]
                self.feature_std[labels] = self.decay_ratio * var + (
                    1 - self.decay_ratio) * self.feature_std[labels]
        else:
            self.feature_mean[labels] = mean
            self.feature_std[labels] = var
            self.feature_used[labels] += 1

    def fa_generate(self):
        rand = torch.rand(self.num_classes).cuda()
        uniq_c = torch.where(rand < self.prob_list)[0]
        embedding_list, label_list = [], []

        if len(uniq_c) > 0:
            for c in uniq_c:
                c = int(c)
                if self.feature_used[c] == 0:
                    continue
                std = torch.sqrt(self.feature_std[c])
                new_sample = self.feature_mean[c] + std * torch.normal(
                    0, 1, size=std.shape).cuda()
                embedding_list.append(new_sample.unsqueeze(0))
                label_list.append(c)

        if len(embedding_list) != 0:
            embedding_list = torch.cat(embedding_list, 0)
            label_list = torch.tensor(label_list, device=embedding_list.device)
            return embedding_list, label_list
        else:
            return [], []

    def dynamic_sampling(self):
        if self.training:
            return

        cum_labels = reduce_mean(self.loss_cls.cum_labels)
        cum_losses = reduce_mean(self.loss_cls.cum_losses)
        sum_loss = cum_labels.sum()
        self.cum_loss_perclass_t1 = nn.Parameter(
            cum_losses / sum_loss, requires_grad=False)

        if self.cum_loss_perclass_t0.sum() == 0:
            self.cum_loss_perclass_t0[:] = self.cum_loss_perclass_t1[:]

        # update group_list
        feature_mean = self.feature_mean
        mean_xy = torch.matmul(feature_mean, feature_mean.T)
        mean_x2 = torch.sum(feature_mean.square(), dim=1).unsqueeze(1)
        mean_y2 = torch.sum(feature_mean.square(), dim=1).unsqueeze(0)
        distance = (mean_x2 - 2 * mean_xy + mean_y2).data.cpu().numpy()
        clustering = AffinityPropagation(random_state=1, affinity='precomputed').fit(distance) # noqa
        n_clustering = max(clustering.labels_)
        self.group_cluster_list = []
        for i in range(n_clustering + 1):
            temp = [ind for ind, v in enumerate(clustering.labels_) if v == i]
            self.group_cluster_list.append(temp)

        # update prob_list
        for i in range(len(self.group_cluster_list)):
            select_group = self.group_cluster_list[i]
            loss_t0 = self.cum_loss_perclass_t0[select_group].sum()
            loss_t1 = self.cum_loss_perclass_t1[select_group].sum()
            delta_value = loss_t1 - loss_t0
            if delta_value > 0:
                self.prob_list[select_group] = (self.prob_list[select_group] *
                                                self.dynamic_down).clamp(0, 1)
            if delta_value < 0:
                self.prob_list[select_group] = (self.prob_list[select_group] *
                                                self.dynamic_up).clamp(0, 1)

        # update loss
        self.cum_loss_perclass_t0[:] = self.cum_loss_perclass_t1[:]

        return

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             embedding=None):
        losses = dict()
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.training:
            pos_embedding = embedding[pos_inds]
            pos_label = labels[pos_inds]
            if len(pos_embedding) > 0:
                self.fa_update(pos_embedding, pos_label)

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        if (self.epoch >= 1) and self.training:
            aug_embedding, aug_labels = self.fa_generate()
            if len(aug_embedding) > 0:
                aug_cls_score = self.fc_cls(
                    aug_embedding) if self.with_cls else None
                aug_label_weights = torch.ones(
                    len(aug_labels), device=aug_embedding.device)
                aug_label_weights = aug_label_weights * self.loss_aug_weight
                aug_avg_factor = max(
                    torch.sum(aug_label_weights > 0).float().item(), 1.)
                loss_cls_aug = self.loss_cls(
                    aug_cls_score,
                    aug_labels,
                    aug_label_weights,
                    avg_factor=aug_avg_factor,
                    reduction_override=reduction_override)
                losses['loss_cls'] += loss_cls_aug
        return losses


@HEADS.register_module()
class Shared2FCFASABBoxHead(ConvFCFASABBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCFASABBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
