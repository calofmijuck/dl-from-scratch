import numpy as np
import pickle
from PIL import Image
from mnist import load_mnist
from activation import softmax
from forward import forward

# normalize field has been set to False to see the incorrect images
def get_data():
    data = load_mnist(normalize=False, flatten=True, one_hot_label=False)
    return data[1]


def load_pretrained():
    with open("sample_weight.pkl", "rb") as f:
        nn = pickle.load(f)
    return nn


def predict(nn, x):
    return softmax(forward(nn, x))


def showImg(img):
    pil = Image.fromarray(np.uint8(img.reshape(28, 28)))
    pil.show()


img, label = get_data()
nn = load_pretrained()

# For showing incorrect images
count = 0
limit = 5

correct = 0
for i in range(len(img)):
    idx = np.argmax(predict(nn, img[i]))
    if idx == label[i]:
        correct += 1
    elif count < limit:
        showImg(img[i])  # Image
        print(idx)  # Prediction
        count += 1


print("Accuracy: " + str(correct / len(img)))
