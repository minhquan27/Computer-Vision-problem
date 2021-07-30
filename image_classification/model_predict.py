from model import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def show_image(link_image):
    img = Image.open(link_image)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = 224, 224

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))

    # Get the dimensions of the new image size
    width, height = img.size

    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img / 255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis, :]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


def predict_image(link_image):
    dict_parameter = torch.load('/Users/nguyenquan/Desktop/mars_project/computer_vision/image_classification'
                                '/model_weight/model_weight.ckpt', map_location=torch.device('cpu'))
    model = Cnn()
    model.load_state_dict(dict_parameter)
    image = process_image(link_image)
    prob = model(image)
    prob = torch.softmax(prob, -1)
    prob = torch.argmax(prob, dim=1).item()
    return prob


if __name__ == '__main__':
    show_image('data/dogs-vs-cats/test/2003.jpg')
    prob = predict_image('data/dogs-vs-cats/test/2003.jpg')
    if prob == 1:
        print("This is a dog!")
    else:
        print("This is a cat!")
