from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.transforms as transforms
import torch


def collate_fn(batch):
    index = [item['index'] for item in batch]
    image = torch.stack([item['image'] for item in batch])
    bbox = [torch.tensor(item['bbox']) for item in batch]
    mask = [torch.stack(item['mask']) for item in batch]
    label = [item['label'] for item in batch]
    
    return {'index': index, 'image': image, 'bbox': bbox, 'mask': mask, 'label': label}



class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]
        return image, masks, bboxes
