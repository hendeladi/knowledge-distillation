import numpy as np
import itertools





def func_1b(b, x):
    out = [0 if i < b else 1 for i in x]
    return np.array(out)


def func_2b(b1, b2, x):
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return np.array(out)


def optimizer_1b(x, y):
    """
    returns the boundary index (0 up to the index exclusive, 1 beyond it).
    Inputs:
        x - sorted x-value vector
        y - corresponding labels
    Outputs:
        b - optimal boundary parameter
        minval - optimal loss on (x,y) obtained with parameter b
    """
    losses = np.array([])
    for i in range(len(x)+1):
        zeros = y[0:i]
        ones = y[i:len(x)]
        zero_loss = sum(zeros) if len(zeros)>0 else 0
        one_loss = np.sum(1 - np.array(ones)) if len(ones)>0 else 0
        loss = one_loss + zero_loss
        losses = np.append(losses, loss)
    minidx = np.argmin(np.array(losses))
    minval = np.min(np.array(losses)/len(losses))
    if minidx == len(x):
        b = x[-1]
    elif minidx == 0:
        b = x[0]
    else:
        b = (x[minidx] + x[minidx - 1]) / 2
    return b, minval


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
    for i, b in enumerate(params):
        hypothesis_labels = func_1b(b, x)
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
    for i, b in enumerate(params):
        hypothesis_labels = func_1b(b, x)
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
    for i, b in enumerate(params):
        hypothesis_labels = func_1b(b, x)
        loss = np.mean(np.abs(hypothesis_labels - np.array(y)))
        losses = np.append(losses, loss)
    minval = np.amin(losses)
    min_indices = np.where(np.isclose(losses, np.amin(losses)))
    min_idx = min_indices[0][0]
    b_opt = params[min_idx]
    return b_opt, minval


class BinaryFunction:
    def __init__(self, parameters):
        if isinstance(parameters, int):
            self.parameters = np.random.uniform(low=0, high=1, size=(parameters,))
        else:
            self.parameters = parameters

    def get_labels(self, x):
        params = np.append(np.append(0, self.parameters), 1)
        left = 0
        zero_or_one = 0
        out = []
        for b in params[1:]:
            right = b
            for feature in x:
                if left <= feature < right:
                    out.append(zero_or_one)
            left = right
            zero_or_one = 1 - zero_or_one
        return np.array(out)

    @staticmethod
    def get_risk(obj1, obj2):
        assert type(obj1) == type(obj2)
        if isinstance(obj1, list) or isinstance(obj1, np.ndarray):
            params1 = np.append(np.append(0, obj1), 1)
            params2 = np.append(np.append(0, obj2), 1)
        elif isinstance(obj1, BinaryFunction):
            params1 = np.append(np.append(0, obj1.parameters), 1)
            params2 = np.append(np.append(0, obj2.parameters), 1)
        else:
            raise "{} is not a valid type".format(type(obj1))

        dict1 = {}
        zero_or_one = 0
        for i,j in zip(params1[:-1],params1[1:]):
            dict1[(round(i,10),round(j,10))] = zero_or_one
            zero_or_one = 1 - zero_or_one

        dict2 = {}
        zero_or_one = 0
        for i,j in zip(params2[:-1],params2[1:]):
            dict2[(round(i,10),round(j,10))] = zero_or_one
            zero_or_one = 1 - zero_or_one
        all_params = sorted(list(set(np.append(params1, params2))))

        def get_closest(val, lst):
            lst_sorted = sorted(lst)
            idx = [i for i,x in enumerate(lst_sorted) if val<x][0]
            right = lst_sorted[idx]
            left = 0 if right == 0 else lst_sorted[idx-1]
            return left, right

        for p in all_params:
            if p not in params1:
                (left, right) = get_closest(p, params1)
                dict1[(round(left, 10), round(p, 10))] = dict1[(round(left, 10), round(right, 10))]
                dict1[(round(p, 10), round(right, 10))] = dict1[(round(left, 10), round(right, 10))]
                params1 = np.append(params1, p)

            if p not in params2:
                (left, right) = get_closest(p, params2)
                dict2[(round(left, 10), round(p, 10))] = dict2[(round(left, 10), round(right, 10))]
                dict2[(round(p, 10), round(right, 10))] = dict2[(round(left, 10), round(right, 10))]
                params2 = np.append(params2, p)
        risk = 0
        for left, right in zip(all_params[:-1],all_params[1:]):
            risk += (right - left)*abs(dict1[(round(left, 10), round(right, 10))] - dict2[(round(left, 10), round(right, 10))])
        return round(risk, 10)

    def get_approx_hypoth(self, num_params):
        if len(self.parameters) <= num_params:
            return 0, self.parameters, True
        opt_params = None
        min_risk = 10
        is_unique = True
        for parameters in itertools.product(self.parameters, repeat=num_params):
            parameters = list(parameters)
            if sorted(parameters) == parameters:
                risk = BinaryFunction.get_risk(parameters, self.parameters)
                if risk == min_risk:
                    is_unique = False
                elif risk < min_risk:
                    is_unique = True
                    min_risk = risk
                    opt_params = parameters
        return min_risk, opt_params, is_unique






