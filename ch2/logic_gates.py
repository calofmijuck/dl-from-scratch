import numpy as np


def AND(a, b):
    x = np.array([a, b])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(a, b):
    x = np.array([a, b])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(a, b):
    x = np.array([a, b])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# A xor B = ~(A and B) and (A or B)
def XOR(a, b):
    t1 = NAND(a, b)
    t2 = OR(a, b)
    return AND(t1, t2)
