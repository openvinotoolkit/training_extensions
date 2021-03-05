import torch

from .utils import AverageMeter, calculate_accuracy, prepare_batch


def validate(args, epoch, data_loader, model, criterion, logger):
    model.eval()

    video_acc = AverageMeter()
    clip_logits = []
    previous_video_id = None
    for i, (inputs, targets) in logger.scope_enumerate(data_loader, epoch, total_time='time/val_epoch',
                                                       fetch_time='time/val_data', body_time='time/val_step'):
        video_ids = targets['video']
        batch_size, inputs, labels = prepare_batch(args, inputs, targets)
        with torch.no_grad():
            outputs = model(*inputs)

        # compute video acc
        for j in range(outputs.size(0)):
            if video_ids[j] != previous_video_id and previous_video_id is not None:
                clip_logits = torch.stack(clip_logits)
                video_logits = torch.mean(clip_logits, dim=0)
                probs, preds = torch.topk(video_logits, k=1)

                video_acc.update((previous_video_gt == preds).item())
                clip_logits = []

            clip_logits.append(outputs[j].data.cpu())
            previous_video_id = video_ids[j]
            previous_video_gt = labels[j].cpu()

        loss = criterion(outputs=outputs, targets=labels, inputs=inputs)
        acc = calculate_accuracy(outputs, labels)

        logger.log_value("val/loss", loss.item(), batch_size)
        logger.log_value("val/acc", acc, batch_size)
    logger.log_value("val/video", video_acc.avg)

    return logger.get_value("val/acc")
