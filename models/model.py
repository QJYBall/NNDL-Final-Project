from einops.einops import rearrange
from utils import mixup_data, rand_bbox
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import Variable
import networks.resnet as resnet


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._names_losses = ('cross entropy loss', )
        if cfg.dataset.startswith('cifar'):
            self._num_classes = int(cfg.dataset[5:])

        dict_nets = {
            'resnet18': resnet.ResNet18,
            'resnet34': resnet.ResNet34,
            'resnet50': resnet.ResNet50,
            'resnet101': resnet.ResNet101,
            'resnet152': resnet.ResNet152
        }
        self.net = dict_nets[self.cfg.network.lower()](self._num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.net(imgs)

    def forward_val(self, imgs):
        return self.net(imgs)

    def preprocessing(self, input, ground_truth, device):
        if self.cfg.mixup:
            images, target_a, target_b, lam = mixup_data(input, ground_truth, device, self.cfg.alpha)
            images, target_a, target_b = map(Variable, (images, target_a, target_b))

            return images, (ground_truth, target_a, target_b, lam)
        elif self.cfg.cutmix:
            self._r = np.random.rand(1)
            if self.cfg.beta > 0 and self._r < self.cfg.cutmix_prob:
                lam = np.random.beta(self.cfg.beta, self.cfg.beta)
                rand_index = torch.randperm(input.size()[0]).to(device)
                target_a = ground_truth
                target_b = ground_truth[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                return input, (ground_truth, target_a, target_b, lam)
            else:
                return input, ground_truth
        else:
            return input, ground_truth

    def total_loss(self, output, ground_truth):
        """total_loss.

        Args:
            output: output of forward(input)
            ground_truth: ground truth from training dataset
        Returns:
            losses: 1-D tensor that requires no gradient; last element is total loss, and others are individual losses
            total_loss: tensor for back propagation, gradient required
        """
        if self.cfg.mixup or (self.cfg.cutmix and self.cfg.beta > 0 and self._r < self.cfg.cutmix_prob):
            ground_truth, target_a, target_b, lam = ground_truth
            total_loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
        else:
            total_loss = self.criterion(output, ground_truth)

        if not self.cfg.mixup and not self.cfg.cutmix:
            pred = torch.max(output.data, 1)[1]
            num_correct_pred = (pred == ground_truth.data).sum().item()
            num_pics = ground_truth.size(0)
            self.num_pics += num_pics
            self.num_correct_pred += num_correct_pred

        losses = torch.tensor((total_loss, )).detach()

        return losses, total_loss

    def metrics_val(self, output, ground_truth):
        """metrics_val.

        Args:
            output: output of forward_val(input)
            ground_truth: ground truth from validation dataset
        Returns: 
            metrics: 1-D tensor that requires no gradient
        """
        pred = torch.max(output.data, 1)[1]
        num_correct_pred = (pred == ground_truth.data).sum().item()
        num_pics = ground_truth.size(0)

        return torch.tensor([num_correct_pred, num_pics])

    def score_val(self, metrics):
        return metrics[0] / metrics[1]

    @property
    def names_losses(self):
        return self._names_losses

    @property
    def names_metrics_val(self):
        return ('accuracy', )
