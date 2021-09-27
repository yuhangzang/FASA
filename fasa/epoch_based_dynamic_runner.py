import time
import torch

from mmcv.runner import EpochBasedRunner
from mmcv.runner.hooks import TensorboardLoggerHook
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class EpochBasedDynamicRunner(EpochBasedRunner):
    """Epoch-based Runner with the Adaptive Sampling algorithm.
    This runner train models epoch by epoch.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')

        # -- begin: add for adaptive sampling
        if hasattr(self.model, 'module'):
            bbox_head = self.model.module.roi_head.bbox_head
        else:
            bbox_head = self.model.roi_head.bbox_head

        if hasattr(bbox_head, 'epoch'):
            bbox_head.epoch = self._epoch
        if (self.rank == 0) and hasattr(bbox_head, 'tf_writer'):
            for hook in self.hooks:
                if isinstance(hook, TensorboardLoggerHook):
                    bbox_head.tf_writer = hook.writer
        # -- end

        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        # -- begin: add for adaptive sampling
        if hasattr(self.model, 'module'):
            bbox_head = self.model.module.roi_head.bbox_head
        else:
            bbox_head = self.model.roi_head.bbox_head
        bbox_head.loss_cls.open_cums()
        # -- end

        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        # -- begin: add for adaptive sampling
        bbox_head.dynamic_sampling()
        bbox_head.loss_cls.close_cums()
        # -- end

        self.call_hook('after_val_epoch')
