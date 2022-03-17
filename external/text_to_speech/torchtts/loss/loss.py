import torch


def duration_loss(predict, gt, lengths):
    l = torch.sum((predict - gt) ** 2) / torch.sum(lengths)
    return l