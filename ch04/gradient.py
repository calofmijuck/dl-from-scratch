import numpy as np

# Central Difference
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]

        x[idx] = tmp + h
        f1 = f(x)

        x[idx] = tmp - h
        f2 = f(x)

        grad[idx] = (f1 - f2) / (2 * h)
        x[idx] = tmp

    return grad


def gradient_descent(f, x, lr=0.01, num_step=100):
    for i in range(num_step):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def f(x):
    return x[0] ** 2 + x[1] ** 2


x0 = np.array([-3.0, 4.0])
print(gradient_descent(f, x0, lr=0.1))
