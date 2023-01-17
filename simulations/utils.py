import os
import numpy as np


def save_fig(fig=None, dir=os.getcwd(), fname="test.png"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    dest = os.path.join(dir, fname)
    fig.savefig(dest)


def save_data(arr, dir=os.getcwd(), fname="test"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    dest = os.path.join(dir, fname)
    np.save(dest, arr)

