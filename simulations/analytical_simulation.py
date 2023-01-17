import matplotlib.pyplot as plt
from src.sets import Region
from src.statistics import multinomial_coeff
import numpy as np
import random

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


def estimate_berry_esseen(start, n, l, p, pp, repeat=1000, step=1):
    assert start > l
    assert p + pp <= 1
    pc = 1 - p - pp
    mu = pp - p
    var = pp+p
    limit = np.exp(mu*l*(1/var))
    limit = np.exp(mu * (mu+2) * l * (1 / var) * 0.5)
    print(limit)
    Y1 = np.array([])
    Y2 = np.array([])

    def ind(x, p, pp):
        y = None
        if x <= pp:
            y = 1
        elif pp < x <= p+pp:
            y = -1
        else:
            y = 0
        return y

    x_range = list(range(start, n + 1,step))
    for num_samples in x_range:
        repeat1 = np.array([])
        repeat2 = np.array([])
        for r in range(repeat):
            repeat1 = np.append(repeat1, np.mean([ind(random.random(), p, pp) for x in range(num_samples)]))
            repeat2 = np.append(repeat2, np.mean([ind(random.random(), p, pp) for x in range(num_samples-l)]))
        Y1 = np.append(Y1, np.mean(repeat1 > 0))
        Y2 = np.append(Y2, np.mean(repeat2 > l/(num_samples-l)))

    ratio = np.divide(Y2, Y1)
    print(ratio[0])
    plt.figure()
    plt.plot(x_range, Y1, x_range, Y2)
    plt.xlabel('n')
    plt.ylabel('probability')
    plt.title(f'probability of picking wrong hypoth p={p}, pp={pp}, l={l}')
    plt.legend(['regular', f'{l}-constraint'])

    plt.figure()
    plt.plot(x_range, ratio)
    plt.xlabel('n')
    plt.ylabel('ratio')
    plt.title(f'ratio between constraint and regular p={p}, pp={pp}, l={l}')
    plt.ylim([0.5, 1.05])
    plt.show()





if __name__ =='__main__':
    estimate_berry_esseen(100, 100, 1, 0.5, 0.2, repeat=800000, step=1)









