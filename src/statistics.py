import numpy as np
from matplotlib import pyplot as plt
import math


def gen_perm(n, k):
    if k == 1:
        return [[n]]
    if n == 0:
        return [[0]*k]
    return [g2 for x in range(n+1) for g2 in [u+[n-x] for u in gen_perm(x, k-1)]]

def discrete_entropy(pr):
    if not math.isclose(sum(pr), 1,abs_tol=0.0000001):
        raise Exception("sum of probabilities must be 1")
    H = 0
    for p in pr:
        if p > 0:
            H -= p*math.log2(p)
    return H



def eval_entropy(density_fun, x_range, k_range):
    """
    Estimates the differential entropy of density_fun using values in x_range
     and different number of examples according to k_range.
    :param density_fun: a function f(x,k) -> s, where x is a feature value,
        k is the number of examples and s is the density at x.
    :param x_range: list of feature values
    :param k_range: list of num of examples (integers)
    :return: list containing the differential entropy for each k value.
    """
    entropy_arr = []
    eps = 0.0000000001
    n = len(x_range)
    delta = 1 / n
    pdf = np.zeros((k_range[-1] + 1, len(x_range)))
    for k in k_range:
        integ = 0
        for i, x in enumerate(x_range):
            pdf[k][i] = density_fun(x, k)
            integ += -delta * (pdf[k][i] * np.log2(pdf[k][i] + eps))
            # integ += delta * (pdf[k][i])
        entropy_arr.append(integ)
    return entropy_arr


def multinomial_coeff(k, n_lst):
    """
    Computes the multinomial coefficient k choose n_lst[:].
    :param k: integer.
    :param n_lst: list of integers.
    :return: float multinomial coefficient.
    """
    assert sum(n_lst) == k
    num = math.factorial(k)
    denom = 1
    for i in n_lst:
        denom *= math.factorial(i)
    return num // denom


def multinom_most_chosen_class_prob(probabilites, N, class_idx=0):
    if not math.isclose(sum(probabilites), 1,abs_tol=0.0000001):
        raise Exception("sum of probabilities must be 1")

    num_classes = len(probabilites)
    permutations = get_class_permutations(num_classes, N)
    S = 0
    for perm in permutations:
        M = 1
        coeff = multinomial_coeff(N, perm)
        for m, p in zip(perm, probabilites):
            M *= p**m
        S += coeff*M
    return S


def get_class_permutations(num_classes,N):

    def foo(idx, upper, max_val, max_idx):
        if idx == max_idx:
            if upper <= max_val:
                return [[upper]]
            else:
                return None

        concat_lst = []
        for i in range(0, min(max_val,upper)+1):
            lst = foo(idx+1, upper-i, max_val, max_idx)
            if lst is None:
                continue
            concat_lst += [[i] + inner_lst for inner_lst in lst]
        return concat_lst

    concat_lst = []
    min_val = math.ceil(N / num_classes)
    for k in range(min_val, N + 1):
        lst = foo(2, upper=N - k, max_val=k, max_idx=num_classes)
        concat_lst += [[k] + inner_lst for inner_lst in lst]
    return concat_lst






def density_kd(x, k):
    """
    computes the density of the KD student with k examples at point x.
    :param x: float. point to evaluate the density.
    :param k: integer. number of examples.
    :return: float. density value
    """
    if x < 0.25:
        s = k * (x ** (k - 1))
    else:
        s = 0
        for m in range(1, k + 1):
            s += math.comb(k, m) * m * ((1 - x) ** (m - 1)) * (0.25 ** (k - m))
    return s


def density_baseline(x, k):
    """
    computes the density of the baseline student with k examples at point x.
    :param x: float. point to evaluate the density.
    :param k: integer. number of examples.
    :return: float. density value
    """
    if x < 0.25:
        s = k * (x ** (k - 1))
    elif x < 0.8:
        s = 0
        for n in range(1, k + 1):
            coeff = 0
            for m in range(0, n + 1):
                if k - m - n < 0:
                    continue
                coeff += multinomial_coeff(k, [m, n, k - m - n]) * (0.25 ** (k - n - m)) * (0.55 ** n) * (0.2 ** m)
            s += coeff * (n / 0.55) * (((0.8 - x) / 0.55) ** (n - 1))
    else:
        s = 0
        for n in range(1, k + 1):
            coeff = 0
            for m in range(0, n):
                if k - m - n < 0:
                    continue
                coeff += multinomial_coeff(k, [m, n, k - m - n]) * (0.25 ** (k - n - m)) * (0.55 ** m) * (0.2 ** n)
            s += coeff * (n / 0.2) * (((x - 0.8) / 0.2) ** (n - 1))

    return s


def expected_risk_kd(k_range):
    exp_risk_arr = []
    for k in k_range:
        sigma = 0
        for m in range(1, k + 1):
            sigma += math.comb(k, m) * m * ((0.25) ** (k - m)) * (
                    - (2 / (m * (m + 1))) * (0.2) ** (m + 1)
                    + (0.2 / m) * (0.75) ** m
                    + (1 / (m * (m + 1))) * (0.75) ** (m + 1)
            )
        exp_risk = ((0.25) ** k) * (0.45 - (k / (k + 1)) * 0.25) + sigma
        exp_risk_arr.append(exp_risk)
    return exp_risk_arr


def expected_risk_baseline(k_range):
    exp_risk_arr = []
    for k in k_range:
        sigma = 0
        for n in range(1, k + 1):
            alpha = 0
            m_range = list(range(min(n + 1, k - n + 1)))
            for m in range(min(n + 1, k - n + 1)):
                alpha += multinomial_coeff(k, (m, n, k - n - m)) * (0.25 ** (k - n - m)) * (0.55 ** n) * (0.2 ** m)

            beta = 0
            for m in range(min(n, k - n + 1)):
                beta += multinomial_coeff(k, (m, n, k - n - m)) * (0.25 ** (k - n - m)) * (0.55 ** m) * (0.2 ** n)

            sigma += alpha * (0.2 + 0.55 / (n + 1)) + beta * (0.55 - 0.2 / (n + 1))

        exp_risk = (0.25 ** k) * (0.45 - 0.25 * (k / (k + 1))) + sigma
        exp_risk_arr.append(exp_risk)
    return exp_risk_arr


