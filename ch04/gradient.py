import numpy as np

# Central Difference
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 1D
def numerical_gradient1D(f, x):
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


# 2D - There should be a better way ...
def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient1D(f, X)

    grad = np.zeros_like(X)
    for idx, x in enumerate(X):
        grad[idx] = numerical_gradient1D(f, x)

    return grad


def gradient_descent(f, x0, lr=0.01, num_step=100):
    x = x0[:]
    for i in range(num_step):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def f(x):
    return x[0] ** 2 + x[1] ** 2

if __name__ == "__main__":
    x0 = np.array([-3.0, 4.0])
    print(gradient_descent(f, x0, lr=0.1))
