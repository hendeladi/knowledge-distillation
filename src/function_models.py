import numpy as np
import itertools


def func_1b(b, x):
    out = [0 if i < b else 1 for i in x]
    return np.array(out)


def func_2b(b1, b2, x):
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return np.array(out)


class BaseFunction:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_labels(self, x):
        pass

    def get_risk(self, func_obj):
        pass

    def get_approx_hypoth(self, param_approx):
        pass

    def get_empirical_risk(self, features, param_approx):
        pass

    def __str__(self):
        return str(self.parameters)


class BinaryFunction:
    def __init__(self, parameters):
        if isinstance(parameters, int):
            self.parameters = np.random.uniform(low=0, high=1, size=(parameters,))
            self.parameters.sort()
        else:
            self.parameters = np.array(parameters)
            self.parameters.sort()

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

    @staticmethod
    def get_empirical_risk(binary_func, features, num_param_approx):
        labels = binary_func.get_labels(features)
        losses = np.array([])
        opt_params_choices = []
        params = np.append(np.append(0, features), 1)
        combs = itertools.chain([(i, i) for i in params],itertools.combinations(params, 2))
        for hypoth_params in combs: #itertools.product(params, repeat=num_param_approx):
            hypoth_params = list(hypoth_params)
            hypoth = BinaryFunction(hypoth_params)
            hypoth_labels = hypoth.get_labels(features)
            loss = np.mean(np.abs(hypoth_labels - labels))
            losses = np.append(losses, loss)
            opt_params_choices.append(hypoth_params)

        min_loss = np.amin(losses)
        min_indices = np.where(np.isclose(losses, np.amin(losses)))[0]
        candidates = [opt_params_choices[i] for i in min_indices]
        param_idx = 0
        while len(candidates) > 1 and param_idx < num_param_approx:
            p = [x[param_idx] for x in candidates]
            max_params = np.where(np.isclose(p, np.amax(p)))[0]
            candidates = [candidates[i] for i in max_params]
            param_idx += 1

        opt_params = np.array(candidates[0])
        return min_loss, opt_params

    def get_approx_hypoth(self, num_params):
        if len(self.parameters) <= num_params:
            return 0, self.parameters, True
        opt_params = None
        min_risk = 10
        is_unique = True
        params = np.append(np.append(0, self.parameters), 1)
        for parameters in itertools.product(params, repeat=num_params):
            parameters = list(parameters)
            if sorted(parameters) == parameters:
                risk = BinaryFunction.get_risk(np.array(parameters), self.parameters)
                if risk == min_risk:
                    is_unique = False
                elif risk < min_risk:
                    is_unique = True
                    min_risk = risk
                    opt_params = parameters
        return min_risk, opt_params, is_unique

    def __str__(self):
        return str(self.parameters)






