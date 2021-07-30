import os
import zipfile
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def zip_file(data_dir):
    with zipfile.ZipFile(os.path.join(data_dir, 'train.zip')) as train_zip:
        train_zip.extractall(data_dir)

    with zipfile.ZipFile(os.path.join(data_dir, 'test1.zip')) as test_zip:
        test_zip.extractall(data_dir)


class Data:
    def __init__(self, data_dir="data/dogs-vs-cats"):
        self.train_list = self.load_data(data_dir, "train")
        self.test_data = self.load_data(data_dir, "test")
        self.train_data, self.valid_data = train_test_split(self.train_list, test_size=0.2)

    def load_data(self, data_dir, data_type):
        data_list = glob.glob(os.path.join(data_dir + '/' + data_type, '*.jpg'))
        return data_list

    def show_image(self, idx):
        fig = plt.figure()
        img = Image.open(self.train_list[idx])
        plt.imshow(img)
        plt.axis('off')
        plt.show()


