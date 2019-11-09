import numpy as np


def AND(a, b):
    x = np.array([a, b])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    return int(tmp > 0)


def NAND(a, b):
    x = np.array([a, b])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    return int(tmp > 0)


def OR(a, b):
    x = np.array([a, b])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    return int(tmp > 0)


# A xor B = ~(A and B) and (A or B)
def XOR(a, b):
    return AND(NAND(a, b), OR(a, b))
