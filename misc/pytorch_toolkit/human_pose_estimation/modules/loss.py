import torch.nn as nn

def l2_loss(input, target, batch_size, mask=None):
    if mask is not None:
        loss = (input - target) * mask
    else:
        loss = (input - target)
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()

def mse_loss(output, target, mask):
    mse = nn.MSELoss()
    batch_size = output.size(0)
    num_keypoints = output.size(1)
    heatmaps_target = target.reshape((batch_size, num_keypoints, -1)).split(1, 1)
    heatmaps_pred = output.reshape((batch_size, num_keypoints, -1)).split(1, 1)
    loss = 0
    for idx in range(num_keypoints):
        heatmap_pred = heatmaps_pred[idx].squeeze()
        heatmap_target = heatmaps_target[idx].squeeze()
        loss += 0.5 * mse(heatmap_pred.mul(mask.cuda()[:, idx]),
                          heatmap_target.mul(mask[:, idx]).cuda())

    return loss / num_keypoints
