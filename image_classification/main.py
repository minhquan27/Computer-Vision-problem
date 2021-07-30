import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from load_data import *
from model import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class dataset(torch.utils.data.Dataset):
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


data_dir = "data/dogs-vs-cats"
d = Data(data_dir)
train_data, valid_data, test_data = d.train_data, d.valid_data, d.test_data
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), ])

train_data = dataset(train_data, transform=transforms)
test_data = dataset(test_data, transform=transforms)
val_data = dataset(valid_data, transform=transforms)


class Experiment:
    def __init__(self, learning_rate, num_epoch, batch_size):
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def train_and_eval(self):
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=True)
        model = Cnn().to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        print("Starting training...")
        for epoch in range(self.num_epoch):
            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in train_loader:
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = ((output.argmax(dim=1) == label).float().mean())
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

            print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in val_loader:
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = ((val_output.argmax(dim=1) == label).float().mean())
                    epoch_val_accuracy += acc / len(val_loader)
                    epoch_val_loss += val_loss / len(val_loader)

                print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_val_accuracy,
                                                                            epoch_val_loss))


if __name__ == '__main__':
    expriment = Experiment(learning_rate=0.001, num_epoch=10, batch_size=100)
    expriment.train_and_eval()
