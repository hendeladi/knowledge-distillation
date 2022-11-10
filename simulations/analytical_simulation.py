import matplotlib.pyplot as plt
from src.sets import Region
from src.statistics import multinomial_coeff
import numpy as np


def calc_term2_given_constraint(n, prob_lst, d):
    p = prob_lst[0]
    pt = prob_lst[1]
    pc = prob_lst[2]
    assert p + pt + pc == 1
    if d > n:
        return 0

    s = 0
    for mt in range(d, n+1):
        for m in range(1 + min(mt-d, n-mt)):
            mc = n-m-mt
            assert m+mt+mc == n
            assert mt >= m + d
            s += multinomial_coeff(n, [m, mt, mc]) * (p**m)*(pt**mt)*(pc**mc)
    return s


def compare_term2_with_constraint(n, prob_lst, d, mode="both"):
    with_constraint = []
    without_constraint = []
    if mode == "even":
        if (d+1)%2 ==0:
            rng = list(range(d+1, n, 2))
        else:
            rng = list(range(d + 2, n, 2))
    elif mode == "odd":
        if (d+1)%2 !=0:
            rng = list(range(d+1, n, 2))
        else:
            rng = list(range(d + 2, n, 2))
    else:
        rng = list(range(d + 1, n))

    for i in rng:
        without_constraint.append(calc_term2_given_constraint(i, prob_lst, 0))
        with_constraint.append(calc_term2_given_constraint(i, prob_lst, d))
    without_constraint = np.array(without_constraint)
    with_constraint = np.array(with_constraint)
    plt.figure(1)
    plt.plot(rng, without_constraint, rng, with_constraint)
    plt.title("Pd with constraints")
    plt.legend(["d=0", f"d={d}"])
    plt.xlabel("num samples")
    plt.ylabel("probability")

    plt.figure(2)
    plt.plot(rng, with_constraint/without_constraint)
    plt.title("Pd with/without constraints ratio")
    plt.legend(["with/without"])
    plt.xlabel("num samples")
    plt.ylabel("ratio")

    plt.show()