import numpy as np
import random
import math
import matplotlib.pyplot as plt
from src.statistics import multinomial_coeff


def H(p_lst):
    return sum([-p * math.log2(p) if p > 0 else 0 for p in p_lst])


def D(p_lst, q_lst):
    return sum([p * math.log2(p / q) if p > 0 else 0 for p, q in zip(p_lst, q_lst)])

def rand_test(n, p, q, repeat=1000):
    def ind(x, p):
        if x <= p:
            y = 1
        else:
            y = -1
        return y

    assert p <= 1

    # limit = np.exp(mu*l*(1/var))
    #limit = np.exp(mu * (mu + 2) * l * (1 / var) * 0.5)

    Y1 = np.array([])
    Y2 = np.array([])

    repeat1 = np.array([])
    repeat2 = np.array([])
    for r in range(repeat):
        repeat1 = np.append(repeat1, np.mean([ind(random.random(), p) for x in range(n)]))
        repeat2 = np.append(repeat2, np.mean([ind(random.random(), q) for x in range(n)]))
    Y1 = np.append(Y1, np.mean(repeat1 > 0))
    Y2 = np.append(Y2, np.mean(repeat2 > 0))
    return Y1[0], Y2[0]


def multinom_l_test(n, l, p1, p2):
    assert p1 < p2
    assert l < n
    pc = 1 - p1 - p2

    s = sum([multinomial_coeff(n, [k1, k2, n-k1-k2]) * (p1 ** k1) * (p2 ** k2) * (pc ** (n - k1 - k2))
             for k1 in range(n+1) for k2 in range(min(k1, n-k1) + 1)])

    s_l = sum([multinomial_coeff(n-l, [k1, k2, n-l-k1-k2]) * (p1 ** k1) * (p2 ** k2) * (pc ** (n - l - k1 - k2))
             for k1 in range(l, n - l  +1) for k2 in range(min(k1-l, n-l-k1) + 1)])
    return s_l/s


def binom_l_test(n, l, p):
    s_l = sum([math.comb(n - l, k) * (p ** k) * ((1 - p) ** (n - l - k)) for k in range(math.ceil((n / 2)), n - l + 1)])
    s = sum([math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(math.ceil((n / 2)), n + 1)])
    return s_l/s


def binom_approx_test(n, p):
    def H(p_lst):
        return sum([-p*math.log2(p) if p>0 else 0 for p in p_lst])
    def D(p_lst, q_lst):
        return sum([p*math.log2(p/q) if p>0 else 0 for p, q in zip(p_lst, q_lst)])


    s = sum([math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(math.ceil((n / 2)), n + 1)])
    #s = math.comb(n, 1+ int(n / 2)) * (p ** (1+n / 2)) * ((1 - p) ** (n - (1+n / 2)))
    #s = sum([(p/(1-p))**k for k in range(0, math.ceil((n / 2))-1)])
    s_approx = sum([((1-(2*k/n)**2)**(n/2)) * ((1-4*(k/n)**2)**(-0.5)) * (p/(1-p))**k for k in range(0, math.ceil((n / 2))-1)])
    #s_approx = math.comb(n, int(n / 2)) * (p ** (n / 2)) * ((1 - p) ** (n - (n / 2)))
    s_approx = math.sqrt(2/(math.pi*n))* ((1-p)/(1-2*p)) * 2**(-n*(D((0.5,0.5), (p,1-p))))
    #s_approx = sum([math.comb(n, k) * 2**(-n*(H((k/n, 1-k/n)) + D((k/n, 1-k/n), (p, 1-p)))) for k in range(math.ceil((n / 2)), n + 1)])
    return s_approx/s ,s ,s_approx


def binom_prob_test(n,p1,p2):
    s1 = sum([math.comb(n, k) * (p1 ** k) * ((1 - p1) ** (n - k)) for k in range(math.ceil((n / 2)), n + 1)])
    s2 = sum([math.comb(n, k) * (p2 ** k) * ((1 - p2) ** (n - k)) for k in range(math.ceil((n / 2)), n + 1)])
    return s1/s2

if __name__ == '__main__':
    test = 'binom_l'
    if test =='binom_l_test':
        N = 620
        l = 2
        p1 = 0.3
        p2 = 0.45
        n_range = list(range(10, N, 10))

        mu = p1 - p2
        var = p1 + p2 - mu**2
        limit = np.exp(mu * (mu + 2) * l * (1 / var) * 0.5)
        print(f'limit = {limit}')

        ratio_lst = [multinom_l_test(n, l, p1, p2) for n in n_range]
        fig = plt.figure()
        plt.plot(n_range, ratio_lst, 'blue')
        plt.axhline(y=limit, color='r', linestyle='dashed')
        plt.xlabel('n')
        plt.ylabel('ratio')
        plt.title(f'ratio between probabilities as function of n \n(l={l}, p1={p1}, p2={p2}, limit={limit:.3f})')
        plt.legend(['computed ratio', '$exp(\mu(\mu+2)l/2\sigma^2)$'])
        plt.show()


    if test == 'binom_prob':
        N = 1000
        p1 = 0.4
        p2 = 0.45
        n_range = list(range(2, N, 10))
        ratio_lst = [binom_prob_test(n, p1, p2) for n in n_range]
        fig = plt.figure()
        plt.plot(n_range, ratio_lst, 'blue')
        plt.xlabel('n')
        plt.ylabel('ratio')
        plt.title(f'ratio between probabilities as function of n \n(p1={p1}, p2={p2})')
        plt.show()

    if test == 'binom_l':
        N = 1000
        p = 0.31
        l = 3

        limit = 2**((l/2)*math.log2(p/(1-p)) + l*D([0.5,0.5],[p, 1-p]))
        n_range = list(range(4, N+1, 2))
        #print(f'limit = {limit}')
        ratio_lst = [binom_l_test(n, l, p) for n in n_range]
        print(limit)
        print(ratio_lst[-1])
        fig = plt.figure()
        plt.plot(n_range, ratio_lst, 'blue')
        plt.axhline(y=limit, color='r', linestyle='dashed')
        plt.xlabel('n')
        plt.ylabel('ratio')
        plt.title(f'ratio between probabilities as function of n \n(l={l}, p={p}, limit={limit})')
        plt.legend(['computed ratio', 'theoretical limit'])
        plt.show()

    if test == 'binom_approx':
        N = 1000
        p = 0.45
        n_range = list(range(4, N+1, 2))

        ratio_lst = [binom_approx_test(n, p)[0] for n in n_range]
        s = [binom_approx_test(n, p)[1] for n in n_range]
        s_approx = [binom_approx_test(n, p)[2] for n in n_range]
        print(ratio_lst[-1])
        fig1 = plt.figure()
        plt.plot(n_range, ratio_lst, 'blue')
        plt.xlabel('n')
        plt.ylabel('ratio')
        plt.title(f'ratio between probabilities as function of n \n(p={p})')
        plt.legend(['estimated ratio'])
        fig2 = plt.figure()
        plt.plot(n_range, s, 'blue', n_range, s_approx, 'red')
        plt.xlabel('n')
        plt.legend(['s', 's_approx'])
        plt.show()


