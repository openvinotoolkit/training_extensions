import torch
from torch import nn


class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """Initialize fixed foreground/background loss balancer
        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(self, loss, gt_boxes2d, num_gt_per_img):
        """Forward pass
        Args:
            loss [torch.Tensor(B, H, W)]: Pixel-wise loss
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = compute_fg_mask(
            gt_boxes2d=gt_boxes2d,
            shape=loss.shape,
            num_gt_per_img=num_gt_per_img,
            downsample_factor=self.downsample_factor,
            device=loss.device,
        )
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        return loss


def compute_fg_mask(gt_boxes2d, shape, num_gt_per_img, downsample_factor=1, device=torch.device("cpu")):
    """Compute foreground mask for images
    Args:
        gt_boxes2d [torch.Tensor(B, N, 4)]: 2D box labels
        shape [torch.Size or tuple]: Foreground mask desired shape
        downsample_factor [int]: Downsample factor for image
        device [torch.device]: Foreground mask desired device
    Returns:
        fg_mask [torch.Tensor(shape)]: Foreground mask
    """
    # ipdb.set_trace()
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
    B = len(gt_boxes2d)
    for b in range(B):
        for n in range(gt_boxes2d[b].shape[0]):
            u1, v1, u2, v2 = gt_boxes2d[b][n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask
