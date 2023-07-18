import torch
import torch.nn.functional as F


def aggregate_loss(loss, weight=None, mask=None):
    if torch.numel(loss) == 0:
        return torch.tensor(0., device=loss.device)
    if mask is not None:
        if mask.sum() == 0:
            return 0.*loss.mean()  # keep loss dependent on prediction
        mask = mask.expand_as(loss)
        if weight is not None:
            weight = weight.expand_as(loss)
        loss = loss[mask]
    if weight is not None:
        if mask is not None:
            weight = weight[mask]
        loss = (loss*weight).sum()/weight.sum()
    else:
        loss = loss.mean()
    return loss


def l1(pred, label=0, weight=None, mask=None):
    loss = (pred.contiguous()-label).abs()
    return aggregate_loss(loss, weight=weight, mask=mask)

def strong_l1(pred, label, dim=-1, weight=None, mask=None, eps=1.e-5):
    mse = (pred.contiguous()-label).abs()
    pred_norm = torch.linalg.norm(pred, ord=1, dim=dim, keepdim=True)
    label_norm = torch.linalg.norm(label, ord=1, dim=dim, keepdim=True)
    norm = 1 / torch.clamp(torch.sqrt(pred_norm * label_norm), min=eps, max=1)
    loss = norm * mse
    return aggregate_loss(loss, weight=weight, mask=mask)

def mse(pred, label=0, weight=None, mask=None):
    loss = (pred.contiguous()-label)**2
    return aggregate_loss(loss, weight=weight, mask=mask)

def norm_mse(pred, label, dim=-1, weight=None, mask=None, eps=1.e-5):
    pred_norm = torch.linalg.norm(pred, dim=dim, keepdim=True)
    label_norm = torch.linalg.norm(label, dim=dim, keepdim=True)
    norm = torch.clamp((pred_norm + label_norm), min=eps)
    loss = ((pred.contiguous()-label) / norm) ** 2
    return aggregate_loss(loss, weight=weight, mask=mask)

def strong_mse(pred, label, dim=-1, weight=None, mask=None, eps=1.e-5):
    mse = (pred.contiguous()-label) ** 2
    pred_norm = torch.linalg.norm(pred, dim=dim, keepdim=True)
    label_norm = torch.linalg.norm(label, dim=dim, keepdim=True)
    norm = 1 / torch.clamp(pred_norm * label_norm, min=eps, max=1)
    loss = norm * mse
    return aggregate_loss(loss, weight=weight, mask=mask)

def bce(pred, label, weight=None, mask=None):
    label = label.expand_as(pred)
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
    return aggregate_loss(loss, weight=weight, mask=mask)

def cos(pred, label, weight=None, mask=None):
    loss = (1-F.cosine_similarity(pred, label, dim=-1)) / 2
    return aggregate_loss(loss, weight=weight, mask=mask)