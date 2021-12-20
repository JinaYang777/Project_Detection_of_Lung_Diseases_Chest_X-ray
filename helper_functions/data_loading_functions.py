# torchd
from helper_dataset import get_dataloaders_cifar10, UnNormalize
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_train import train_model
from helper_evaluation import set_all_seeds, set_deterministic, compute_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)

# other
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import warnings
from skimage import io
# from PIL import Image
import time

warnings.filterwarnings("ignore")


# dataset_path = '.\\Combined_data' ### change this

def train_val_dataset(dataset):
    np.random.seed(1)
    num = np.arange(len(dataset))
    np.random.shuffle(num)

    train_idx, test_idx, val_idx = np.split(
        num, [int(.8*len(dataset)), int(.9*len(dataset))])  # len(dataset) didn't work -> 21165

    return train_idx, test_idx, val_idx


def split_train_test_loaders(dataset_path='.\\COVID-19_Radiography_Dataset', batch_size=64):
    np.random.seed(1)

#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.2255)
    mean = [0.4363, 0.4328, 0.3291]
    std = [0.2129, 0.2075, 0.2038]
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # train - test - validation split
    test_dataset = ImageFolder(dataset_path, transform=test_transforms)
    train_dataset = ImageFolder(dataset_path, transform=train_transforms)
    train_idx, test_idx, val_idx = train_val_dataset(test_dataset)

    test_data = Subset(test_dataset, test_idx)
    train_data = Subset(train_dataset, train_idx)
    val_data = Subset(train_dataset, val_idx)

    # loading data
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return (train_data, val_data, test_data), train_loader, valid_loader, test_loader


def convert_lab(lab):
    label_dc = {0: 'COVID', 1: 'Lung_Opacity',
                2: 'Normal', 3: 'Viral Pneumonia'}
    return [label_dc[int(i)] for i in list(lab)]


def show_img(dataset_path='.\\COVID-19_Radiography_Dataset', train=False):
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.2255)
    mean = [0.4363, 0.4328, 0.3291]
    std = [0.2129, 0.2075, 0.2038]
    
    if train:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
        data = ImageFolder(dataset_path, transform=train_transforms)
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        data = ImageFolder(dataset_path, transform=data_transform)

    loader = DataLoader(data, 4, shuffle=True)

    batch = next(iter(loader))
    img, lab = batch

    labels = convert_lab(lab)
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))

    for ind, image in enumerate(img):
        axes[ind].imshow(image.permute(1, 2, 0))
        axes[ind].set_title(labels[ind])

    # fig.savefig('display_images.jpg')


def show_img_orig(data):
    loader = DataLoader(data, 4, shuffle=True)

    batch = next(iter(loader))
    img, lab = batch

    grid = torchvision.utils.make_grid(img, nrow=4)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    print(f'actual labels: {convert_lab(lab)}')
    print(f'numerical labels: {lab}')
