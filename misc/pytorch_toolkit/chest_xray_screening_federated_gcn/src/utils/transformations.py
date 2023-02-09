from torchvision import transforms
################# Data Augmentation and Transforms #####################

# Training Transformations/ Data Augmentation
train_transform=transforms.Compose([
                                    transforms.Resize(350),
                                    transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
# Test/Val Transformations
test_transform=transforms.Compose([
                                    transforms.Resize(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
                                    