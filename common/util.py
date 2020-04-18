import numpy as np


def shuffle_dataset(x, t):
    # Shuffle dataset. x is the data, t is the label
    p = np.random.permutation(x.shape[0])
    if x.ndim == 2:
        x = x[p, :]
    else:
        x = x[p, :, :, :]
    
    return x, t[p]
