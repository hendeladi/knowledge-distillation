import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt


class MultiRepeat:
    def __init__(self, n_processes, n_repeats, worker_func, reduction_func=np.mean):
        self.n_processes = n_processes
        self.repeats_per_core = n_repeats // n_processes
        self.reduction_func = reduction_func
        self.worker_func = worker_func

    def run(self, n, p, l):
        manager = multiprocessing.Manager()
        return_results = manager.list()
        jobs = []
        for i in range(self.n_processes):
            pid = multiprocessing.Process(target=self.worker_func, args=(self.repeats_per_core, n, p, l, return_results))
            jobs.append(pid)
            pid.start()
        for proc in jobs:
            proc.join()
        outputs = []
        for i in range(self.n_processes):
            outputs.append(return_results[i])
        result = self.reduction_func(outputs)
        return result




def rand_test(n, p, l, repeat=1000):
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
        repeat2 = np.append(repeat2, np.mean([ind(random.random(), p) for x in range(n - l)]))
    Y1 = np.append(Y1, np.mean(repeat1 > 0))
    Y2 = np.append(Y2, np.mean(repeat2 > l / (n - l)))
    return Y1[0], Y2[0]


if __name__ == '__main__':
    n_processes = 40
    n_repeats = 100000000
    n_range = list(range(500, 501, 5))
    dest = "/local/mnt/workspace/ahendel/ratio.png"


    def reduction_func(lst):
        Y1 = []
        Y2 = []
        for pair in lst:
            Y1.append(pair[0])
            Y2.append(pair[1])
        Y1_mean = np.mean(Y1)
        Y2_mean = np.mean(Y2)
        ratio = np.divide(Y2_mean, Y1_mean)
        return ratio


    def worker_func(repeatitions, n, p, l, return_results):
        Y1, Y2 = rand_test(n, p, l, repeatitions)
        return_results.append([Y1, Y2])


    res_arr = []
    p = 0.4
    l = 1
    mu = 2 * p - 1
    var = p + 1 - p - mu ** 2
    limit = np.exp(mu * (mu + 2) * l * (1 / var) * 0.5)

    for n in n_range:
        multi = MultiRepeat(n_processes, n_repeats, worker_func, reduction_func)
        res = multi.run(n=n, p=p, l=l)
        res_arr.append(res)
        print(f'n = {n}, ratio = {res}')
    print(f'ratio = {res_arr[0]}')
    fig = plt.figure()
    plt.plot(n_range, res_arr, 'blue')
    plt.axhline(y=limit, color='r', linestyle='dashed')
    plt.xlabel('n')
    plt.ylabel('ratio')
    plt.title(f'ratio between probabilities as function of n \n(repetitions={n_repeats}, l={l}, p={p})')
    plt.legend(['estimated ratio', '$exp(\mu(\mu+2)l/2\sigma^2)$'])
    fig.savefig(dest)







