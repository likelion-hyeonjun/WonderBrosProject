'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from deformable_filter import *

import os
import argparse

from models import *
from utils import fineTuningModel
from train import train
from test import Test

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#오늘 모델돌릴수있는데까지 짜고 내일 state이거해야겠다
parser = argparse.ArgumentParser(description='GroceryDataset Training with VGG, ResNet and DenseNet')
parser.add_argument('--model', 
                    default ='vgg19',
                    help='choose model for experiment \n available params: densenet, resnet, vgg19')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--train_batch_size', type=int, default=1, help='# images in batch when training')
parser.add_argument('--test_batch_size', type=int, default=1, help='# images in batch when testing')
parser.add_argument('--opt', default='SGD', help='optimizer')
parser.add_argument('--freeze', type=str2bool ,default="true", help='true: fine-tuning for only classifier layer, false: fine-tuning for whole layer (with pretrained)')

args = parser.parse_args()

# 추가해야할 부분 fint turning 관련된 arg, data augmentation 관련 arg

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DataSet
print('==> Preparing data..')
preprocess_densenet = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

preprocess_vgg = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

preprocess_resnet = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if 'vgg' in args.model :
    preprocess = preprocess_vgg
elif 'resnet' in args.model :
    preprocess = preprocess_resnet
elif 'densenet' in args.model :
    preprocess = preprocess_densenet

trainDataPath = trainDataPath = os.getcwd()+"/GroceryStoreDataset-master/dataset/train/Packages"
train_dataset = ImageFolder(root=trainDataPath, transform = preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.train_batch_size,
                                            shuffle = True)
testDataPath = os.getcwd()+"/GroceryStoreDataset-master/dataset/test/Packages"
test_dataset = ImageFolder(root=testDataPath, transform = preprocess) #이거 test시에해야하나?
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = args.test_batch_size,
                                          shuffle = False)
validDataPath = os.getcwd()+"/GroceryStoreDataset-master/dataset/val/Packages"
valid_dataset = ImageFolder(root=validDataPath, transform = preprocess)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                          batch_size = args.test_batch_size,
                                          shuffle = False)
# Model
print('==> Building model..')

classes = ('Alpro-Blueberry-Soyghurt','Alpro-Fresh-Soy-Milk',
            'Alpro-Shelf-Soy-Milk', 'Alpro-Vanilla-Soyghurt',
            'Arla-Ecological-Medium-Fat-Milk', 'Arla-Ecological-Sour-Cream',
            'Arla-Lactose-Medium-Fat-Milk', 'Arla-Medium-Fat-Milk',
            'Arla-Mild-Vanilla-Yoghurt', 'Arla-Natural-Mild-Low-Fat-Yoghurt',
            'Arla-Natural-Yoghurt','Arla-Sour-Cream',
            'Arla-Sour-Milk','Arla-Standard-Milk',
            'Bravo-Apple-Juice', 'Bravo-Orange-Juice',
            'Garant-Ecological-Medium-Fat-Milk', 'Garant-Ecological-Standard-Milk',
            'God-Morgon-Apple-Juice', 'God-Morgon-Orange-Juice',
            'God-Morgon-Orange-Red-Grapefruit-Juice', 'God-Morgon-Red-Grapefruit-Juice',
            'Oatly-Natural-Oatghurt', 'Oatly-Oat-Milk',
            'Tropicana-Apple-Juice', 'Tropicana-Golden-Grapefruit',
            'Tropicana-Juice-Smooth', 'Tropicana-Mandarin-Morning',
            'Valio-Vanilla-Yoghurt', 'Yoggi-Strawberry-Yoghurt',
            'Yoggi-Vanilla-Yoghurt')


#resume일 경우?
model = fineTuningModel(args.model, len(classes), args.freeze, True) #is freeze, pretrained 넣어주기

#use deformable !!should modify
# if args.model == 'resnet101':
#     #denseblock 4, denselayer16, last conv filter #model layer 0은 stride 2로 되있어서 안됨. ㅅㅂ
#     model.layer4[1].conv2 = deformable_filter(512,512)
#     model.layer4[2].conv2 = deformable_filter(512,512)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#그냥 트레인으로만들자..  early stopping 넣어서
#해야할거 validation set만들기 그리고 train에 early stopping 하기. 
# train때 load state dict하기 


trained_model = train(model = model, 
                    train_loader = train_loader, 
                    valid_loader = valid_loader,
                    criterion = criterion, 
                    optimizer = optimizer, 
                    scheduler = exp_lr_scheduler, 
                    device = device, 
                    num_train_data =len(train_dataset), 
                    num_valid_data = len(valid_dataset), 
                    num_epochs = args.epoch)
test_model = Test(trained_model, test_loader, len(test_dataset))
test_model.OverallAccuracy()
test_model.ClassAccuracy(classes)