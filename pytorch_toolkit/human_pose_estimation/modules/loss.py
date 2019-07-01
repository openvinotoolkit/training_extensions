def l2_loss(input, target, batch_size, mask=None):
    if mask is not None:
        loss = (input - target) * mask
    else:
        loss = (input - target)
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()
