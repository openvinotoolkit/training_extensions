import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import torch.nn.functional as F
from dataset_eye import EyeDataset
import torch.onnx
import cv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 50, kernel_size=3)
        self.conv4 = nn.Conv2d(50, 2, kernel_size=1, bias=False, padding=0, stride=1)
        self.max_pool2d = nn.MaxPool2d((4,4))
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2d(x)
        x = self.softmax(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Open close eyes classifier')
    parser.add_argument('data_root')
    parser.add_argument('epoch', type=int)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('--pretrained')
    args = parser.parse_args()
    return args

def train(net, loader, device, optimizer):
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for i, (inputs, labels, _) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).view(len(labels), 2)
       
        loss =  criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss
        # if i % 1 == 9:    # print every 20 mini-batches
        print('iter: %5d loss: %.3f' % (i + 1, running_loss / 10))
        running_loss = 0.0

def test(net, loader, device):
    corrected = 0.
    total = 0.
    with torch.no_grad():
        for i, (inputs, labels, fname) in enumerate(loader):           
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for l,p in zip(labels,predicted):
                if l == p[0][0]:
                    corrected += 1 
                total += 1

    print ("Test accuracy: {}".format(corrected / total))

def main():
    args = parse_args()
    torch.manual_seed(100)
    device = torch.device("cuda")
    net = Net().to(device)
    if args.pretrained :
        net.load_state_dict(torch.load(args.pretrained))
        

    lr = args.lr
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train_transforms = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [1, 1, 1], inplace=False),
                        ])
    test_transfroms = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [1, 1, 1], inplace=False),
                        ])

    train_db = EyeDataset(args.data_root, 'train', train_transforms)
    test_db = EyeDataset(args.data_root, 'val', test_transfroms)

    train_db_loader = torch.utils.data.DataLoader(train_db, batch_size=args.batch_size, shuffle=True,num_workers = 5)
    test_db_loader = torch.utils.data.DataLoader(test_db, batch_size=1, num_workers = 1)

    for epoch in range(args.epoch):         
        if epoch in [10,15,25]:
            lr = 0.1 * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print ('epoch={}, lr={}'.format(epoch,lr))
        train(net, train_db_loader, device, optimizer)
        test(net,test_db_loader,device)

        torch.save(net.state_dict(), "open_close_eyes_epoch_{}.pth".format(epoch))
        dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False).to(device)
        torch_out = torch.onnx.export(net, dummy_input, "open_close_eyes_epoch_{}.onnx".format(epoch), export_params=True)

    pass


if __name__ == '__main__':
    main()
