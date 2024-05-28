import numpy as np
import torch
import torch.nn.functional as F
import torchvision


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, y):
        loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=x,
            targets=y,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss


class FocalLossBCE(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(inputs, targets)
        combined_loss = self.bce_weight * bce_loss + self.focal_weight * focal_loss
        return combined_loss


class FocalCosineLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=0.1, reduction="mean"):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent
        self.reduction = reduction

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(
            input,
            F.one_hot(target, num_classes=input.size(-1)),
            self.y,
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


class LDAM(torch.nn.Module):
    """Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss

    https://arxiv.org/abs/1906.07413
    """

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAM, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LabelSmoothingCrossEntropy(torch.nn.Module):

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(torch.nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class JsdCrossEntropy(torch.nn.Module):
    """Jensen-Shannon Divergence + Cross-Entropy Loss

    Based on implementation:
    https://github.com/google-research/augmix/blob/master/imagenet.py

    From paper: AugMix: A Simple Data Processing Method to Improve Robustness and
    Uncertainty https://arxiv.org/abs/1912.02781
    """

    def __init__(self, num_splits=3, alpha=12, smoothing=0.1):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, target):
        split_size = output.shape[0] // self.num_splits
        assert split_size * self.num_splits == output.shape[0]
        logits_split = torch.split(output, split_size)

        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss += (
            self.alpha
            * sum(
                [
                    F.kl_div(logp_mixture, p_split, reduction="batchmean")
                    for p_split in probs
                ]
            )
            / len(probs)
        )
        return loss
