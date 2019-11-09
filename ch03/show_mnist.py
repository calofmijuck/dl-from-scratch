import sys, os

sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image


def show(img):
    pil = Image.fromarray(np.uint8(img))
    pil.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

show(img)
