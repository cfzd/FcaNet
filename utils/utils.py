import math
import torch
import torch.nn as nn


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters <= self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            if self.iters == self.warmup_iters:
                self.iters = 0
                self.warmup = None
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_v1(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # import pdb; pdb.set_trace()
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
    def forward_v2(self, inputs, targets):
        probs = self.logsoftmax(inputs)
        targets = torch.zeros(probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = nn.KLDivLoss()(probs, targets)
        return loss

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        return self.forward_v1(inputs, targets)
    
    def test(self):
        
        inputs = torch.randn(2, 5)
        targets = torch.randint(0, 5, [2])
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        
        print((targets * torch.log(targets) - targets * log_probs).sum(-1).mean())
        print(nn.KLDivLoss(reduce='mean')(log_probs, targets))
        # import pdb; pdb.set_trace()

class AverageMeter:
    def __init__(self):
        self.reset()

    def update(self, val):
        self.val += val.mean(0)
        self.num += 1
    
    def reset(self):
        self.num = 0
        self.val = 0
    
    def avg(self):
        try:
            return self.val/self.num
        except Exception:
            return None


if __name__ == '__main__':
    criterion = CrossEntropyLabelSmooth(5)
    criterion.test()