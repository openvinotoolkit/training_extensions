import torch
import torch.nn.functional as F

from .utils import AverageMeter, prepare_batch
from .utils import calculate_accuracy


def test(args, data_loader, model, logger):
    print('test')
    model.eval()

    video_acc = AverageMeter()

    output_buffer = []
    previous_video_id = None
    for i, (inputs, targets) in logger.scope_enumerate(data_loader):
        video_ids = targets['video']
        batch_size, inputs, labels = prepare_batch(args, inputs, targets)

        outputs = model(*inputs)

        if args.softmax_in_test:
            outputs = F.softmax(outputs)

        for j in range(outputs.size(0)):

            if video_ids[j] != previous_video_id and not (i == 0 and j == 0):
                # Computed all segments for current video
                video_outputs = torch.stack(output_buffer)
                video_result = torch.mean(video_outputs, dim=0)
                probs, preds = torch.topk(video_result, k=1)

                is_correct_match = (video_gt.cpu() == preds).item()
                video_acc.update(is_correct_match)

                output_buffer.clear()

            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = video_ids[j]
            video_gt = labels[j]

        clip_acc = calculate_accuracy(outputs, labels)

        logger.log_value("test/acc", clip_acc, batch_size)
        logger.log_value("test/video", video_acc.avg)

    return logger.get_value("test/video"), logger.get_value("test/acc")
