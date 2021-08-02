# data augmentation using pytorch and torchvision
# link https://pytorch.org/vision/stable/transforms.html

import numpy as np  # linear algebra
import pandas as pd  # data processing, csv file
import os
import glob
from PIL import Image  # image manipulation

# data visualization
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib

# import pytorch framework
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


# show image by matplotlib
def show_img(path):
    fig = plt.figure()
    img = Image.open(path)
    plt.title('title image')
    plt.axis('off')
    plt.imshow(img)
    plt.show()


# torchvision.transforms: batch Tensor Image (B, C, H, W) with B: batch, C: number channels, H and W: height and width
''' torchvision.transforms.Compose(transforms): Composes several transforms together
     |- torchvision.transforms.CenterCrop(size): Crops the given image at the center.
     |- torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0): 
          Randomly change the brightness, contrast, saturation and hue of an image.
     |- torchvision.transforms.FiveCrop(size): Crop the given image into four corners and the central crop.
     |- torchvision.transforms.Grayscale(num_output_channels=1): Convert image to grayscale.
     |- torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'): Pad the given image on all sides with the given “pad” value
     |- torchvision.transforms.RandomAffine: Random affine transformation of the image keeping center invariant
     |- torchvision.transforms.Resize: Resize the input image to the given size.
     |- torchvision.transforms.Normalize(mean, std, inplace=False): Normalize a tensor image with mean and standard deviation. 
'''

'''------------------------------------------------------------------------------------------------------------------'''
# dataset dog-vs-cats: https://www.kaggle.com/c/dogs-vs-cats
path = "/Users/nguyenquan/Desktop/mars_project/computer_vision/image_classification/data/dogs-vs-cats"
# print(os.listdir(path))
train_list = glob.glob(os.path.join(path + '/train', '*.jpg'))
test_list = glob.glob(os.path.join(path + '/test', '*.jpg'))
print("train list: {} images ".format(len(train_list)), "test list: {} images".format(len(test_list)))

transforms_data = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), ])


class dataset_dog_cat(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    # dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    # load an one of images
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0

        return img_transformed, label


train_data = dataset_dog_cat(train_list, transform=transforms_data)
test_data = dataset_dog_cat(test_list, transform=transforms_data)
# Data loader with batch_size  = 128
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

'''------------------------------------------------------------------------------------------------------------------'''
# dataset fashion MNIST: https://www.kaggle.com/zalando-research/fashionmnist
path = "/Users/nguyenquan/Desktop/mars_project/computer_vision/data/fashion_mnist"
print(os.listdir(path))
train_csv = pd.read_csv(path + "/fashion-mnist_train.csv")
test_csv = pd.read_csv(path + "/fashion-mnist_test.csv")


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        label = []
        image = []

        for i in self.fashion_MNIST:
            # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))
img, label = train_set[0]
print(img.shape)
print(label)
plt.title(output_label(label))
plt.axis("off")
plt.imshow(img.squeeze(), cmap="gray")
plt.show()

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(train_set, batch_size=100)
