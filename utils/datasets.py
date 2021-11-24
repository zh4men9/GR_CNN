# coding=utf-8
# Author: Little-Chen
# Emial: Chenxiuyan_t@163.com

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_data(args):

    ROOT_TRAIN = args.data_path + '/train'
    ROOT_TEST = args.data_path + '/test'

    # 固定到[-1.0, 1.0]范围内
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)), # 裁剪为32 32
        transforms.RandomVerticalFlip(), # 随机垂直翻转
        transforms.RandomHorizontalFlip(), #随机水平翻转
        transforms.RandomRotation(60), #随机-60 - 60°翻转
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5), #随机颜色变化，第一个参数就是亮度的比例，第二个是对比度，第三个是饱和度
        transforms.ToTensor(), # 将0-255范围的像素转为0-1.0范围的tensor
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize])

    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

    return train_dataset,test_dataset

def get_dataloder(train_dataset,test_dataset, args):

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False)

    return train_dataloader,test_dataloader

