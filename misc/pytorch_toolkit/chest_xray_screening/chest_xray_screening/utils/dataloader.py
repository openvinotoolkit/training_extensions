from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path


class RSNADataSet(Dataset):
    def __init__(self, image_list, label_json, image_directory, transform=True):

        image_directory = Path(image_directory)
        image_names = [Path.joinpath(image_directory, x) for x in image_list]
        self.image_names = image_names
        self.labels = label_json

        if transform is not None:
            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([124.978, 124.978, 124.978], [10.868, 10.868, 10.868])
                                        ])

        self.transform = transform


    def __getitem__(self, index):

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        image_id = str(image_name).rsplit('/', maxsplit = 1)[-1]
        label = self.labels[image_id]

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([label])

    def __len__(self):
        return len(self.image_names)
