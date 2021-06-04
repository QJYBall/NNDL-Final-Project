import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from utils import Cutout, rand_bbox, mixup_data, CSVLogger
from resnet import *


parser = argparse.ArgumentParser(description='NNDL Final Project')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')

parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
parser.add_argument('--cutmix', action='store_true', default=False, help='apply cutmix')
parser.add_argument('--mixup', action='store_true', default=False, help='apply mixup')

parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16, help='length of the holes')

parser.add_argument('--beta', type=float, default=1.0, help='parameter for cutmix')
parser.add_argument('--cutmix_prob', type=float, default=1.0, help='parameter for cutmix')

parser.add_argument('--alpha', type=float, default=1.0, help='parameter for mixup')

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=233, help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

if args.cutout == True:
    file_name = 'cutout'
elif args.cutmix == True:
    file_name = 'cutmix'
elif args.mixup == True:
    file_name = 'mixup' 
else:
    file_name = 'none'  


if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# Image Preprocessing
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])


test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])

# dataset
num_classes = 10
train_dataset = datasets.CIFAR10(root='data/', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(root='data/', train=False, transform=test_transform, download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

model = ResNet18(num_classes=num_classes)

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(model_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/cifar_' + file_name + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc


def simple_train():
    
    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()
            pred = model(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            model_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step()

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

    torch.save(model.state_dict(), 'checkpoints/cifar_' + file_name + '.pt')
    csv_logger.close()
    


def cutout_train():
    
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    
    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()
            pred = model(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            model_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step()

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

    torch.save(model.state_dict(), 'checkpoints/cifar_' + file_name + '.pt')
    csv_logger.close()


def cutmix_train():

    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()

            r = np.random.rand(1)

            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                """1.设定lamda的值，服从beta分布"""
                lam = np.random.beta(args.beta, args.beta)
                """2.找到两个随机样本"""
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index] #batch中的某一张
                """3.生成剪裁区域B"""
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                """5.根据剪裁区域坐标框的值调整lam的值"""
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                """6.将生成的新的训练样本丢到模型中进行训练"""
                pred = model(images)
                """7.按lamda值分配权重"""
                loss = criterion(pred, target_a) * lam + criterion(pred, target_b) * (1. - lam)
            else:
                pred = model(images)
                loss = criterion(pred, labels)

            loss.backward()
            model_optimizer.step()

            xentropy_loss_avg += loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step()

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

    torch.save(model.state_dict(), 'checkpoints/cifar_' + file_name + '.pt')
    csv_logger.close()
    

def mixup_train():

    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()

            images, target_a, target_b, lam = mixup_data(images, labels, args.alpha)
            images, target_a, target_b = map(Variable, (images, target_a, target_b))

            pred = model(images)

            xentropy_loss = lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)
            xentropy_loss.backward()
            model_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

        test_acc = test(test_loader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step()

        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

    torch.save(model.state_dict(), 'checkpoints/cifar_' + file_name + '.pt')
    csv_logger.close()


if args.cutout == True:
    cutout_train()
elif args.cutmix == True:
    cutmix_train()
elif args.mixup == True:
    mixup_train()
else:
    simple_train()
