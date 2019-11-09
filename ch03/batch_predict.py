import numpy as np
import pickle
from PIL import Image
from mnist import load_mnist
from activation import softmax
from forward import forward


def get_data():
    data = load_mnist(normalize=True, flatten=True, one_hot_label=False)
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

batch_size = 100
correct = 0
for i in range(0, len(img), batch_size):
    batch_prediction = predict(nn, img[i : i + batch_size])
    idx = np.argmax(batch_prediction, axis=1)
    correct += np.sum(idx == label[i : i + batch_size])

print("Accuracy: " + str(correct / len(img)))
