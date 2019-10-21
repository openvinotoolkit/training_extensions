import torch
import torch.nn as nn

def l2_loss(input, target, batch_size, mask=None):
    if mask is not None:
        loss = (input - target) * mask
    else:
        loss = (input - target)
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()


def mse_loss(input, target, mask=None):
    mse = nn.MSELoss(reduction='elementwise_mean')
    batch_size = input.size(0)
    num_keypoints = input.size(1)
    heatmaps_pred = input.reshape((batch_size, num_keypoints, -1)).split(1, 1)
    heatmaps_gt = target.reshape((batch_size, num_keypoints, -1)).split(1, 1)
    loss = 0

    for idx in range(num_keypoints):
        prediction = heatmaps_pred[idx].squeeze()
        prediction_tmp = torch.zeros(tuple(prediction.shape), dtype=torch.float32)
        gt = heatmaps_gt[idx].squeeze()
        gt_tmp = torch.zeros(tuple(gt.shape), dtype=torch.float32)
        if mask is not None:
            for i in range(batch_size):
                prediction_tmp[i, :] = prediction[i, :] * mask[i, idx]
                gt_tmp[i, :] = gt[i, :] * mask[i, idx]
            loss += 0.5 * mse(prediction_tmp.cuda(), gt_tmp.cuda())
        else:
            loss += 0.5 * mse(prediction, gt_tmp.cuda())

    return loss / num_keypoints