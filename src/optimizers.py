import numpy as np
from src.function_models import BinaryFunction
import itertools

def optimizer_1b_subopt(x, y):
    """
    returns the boundary index (0 up to the index exclusive, 1 beyond it).
    Inputs:
        x - sorted x-value vector. type can be list or array
        y - corresponding labels. type can be list or array
    Outputs:
        b - optimal boundary parameter
        minval - optimal loss on (x,y) obtained with parameter b
    """
    losses = np.array([])
    params = np.append(np.append(0, x), 1)
    func_1b = BinaryFunction(1)
    for i, b in enumerate(params):
        func_1b.parameters = [b]
        hypothesis_labels = func_1b.get_labels(x)
        loss = np.mean(np.abs(hypothesis_labels - np.array(y)))
        losses = np.append(losses, loss)
    minval = np.min(losses)
    minidx = np.argmin(losses)
    b_opt = params[minidx]
    return b_opt, minval

def optimizer_1b_sticky(x, y, margin=1e-8):
    """
    returns the boundary index (0 up to the index exclusive, 1 beyond it).
    Inputs:
        x - sorted x-value vector. type can be list or array
        y - corresponding labels. type can be list or array
    Outputs:
        b - optimal boundary parameter
        minval - optimal loss on (x,y) obtained with parameter b
    """
    losses = np.array([])
    params = np.append(np.append(0, x), 1)
    func_1b = BinaryFunction(1)
    for i, b in enumerate(params):
        func_1b.parameters = [b]
        hypothesis_labels = func_1b.get_labels(x)
        loss = np.mean(np.abs(hypothesis_labels - np.array(y)))
        losses = np.append(losses, loss)
    minval = np.min(losses)
    minidx = np.argmin(losses)
    b_opt = params[minidx]
    if b_opt == 0:
        b_opt = x[0]
    elif b_opt == 1:
        b_opt = x[-1] + margin
    return b_opt, minval



def optimizer_1b_min(x, y):
    """
    returns the boundary index (0 up to the index exclusive, 1 beyond it).
    Inputs:
        x - sorted x-value vector. type can be list or array
        y - corresponding labels. type can be list or array
    Outputs:
        b - optimal boundary parameter
        minval - optimal loss on (x,y) obtained with parameter b
    """
    losses = np.array([])
    params = np.append(np.append(0, x), 1)
    func_1b = BinaryFunction(1)
    for i, b in enumerate(params):
        func_1b.parameters = [b]
        hypothesis_labels = func_1b.get_labels(x)
        loss = np.mean(np.abs(hypothesis_labels - np.array(y)))
        losses = np.append(losses, loss)
    minval = np.amin(losses)
    min_indices = np.where(np.isclose(losses, np.amin(losses)))
    min_idx = min_indices[0][0]
    b_opt = params[min_idx]
    return b_opt, minval




def optimizer_binary_min(x, y, num_params):
    """
    returns the boundary index (0 up to the index exclusive, 1 beyond it).
    Inputs:
        x - sorted x-value vector. type can be list or array
        y - corresponding labels. type can be list or array
    Outputs:
        b - optimal boundary parameter
        minval - optimal loss on (x,y) obtained with parameter b
    """
    losses = np.array([])
    params = np.append(np.append(0, x), 1)
    for parameters in itertools.product(self.parameters, repeat=num_params):
        parameters = list(parameters)











    func_1b = BinaryFunction(1)
    for i, b in enumerate(params):
        func_1b.parameters = [b]
        hypothesis_labels = func_1b.get_labels(x)
        loss = np.mean(np.abs(hypothesis_labels - np.array(y)))
        losses = np.append(losses, loss)
    minval = np.amin(losses)
    min_indices = np.where(np.isclose(losses, np.amin(losses)))
    min_idx = min_indices[0][0]
    b_opt = params[min_idx]
    return b_opt, minval