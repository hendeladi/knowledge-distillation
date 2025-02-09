import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sets import Region, SecondTerm
from st_simulation import Simulation
from configs import CONFIGS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config to use from config dict')
    parser.add_argument('--dest_dir', type=str, default='./', help='path to save results')
    parser.add_argument('--n_cores', type=int, default=1, help='number of cpu cores to use')
    args = parser.parse_args()

    conf = CONFIGS['example1']
    sim = Simulation(conf, multi_proc=args.n_cores)
    x_axis, baseline_metrics = sim.run()
    baseline_main_term = baseline_metrics['main_term']

    D = [Region([(0.6, 0.9)])]
    Dp = [Region([(0.9, 1)])]
    n_range = list(range(4, 120, 2))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = baseline_main_term[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)
##################################################################
    ell = 2
    a = SecondTerm(D, Dp)
    exponent_arr = np.array([])
    for n in n_range:
        p = 1 - a.probability(n)
        exponent = -(1 / n) * np.log(p)
        exponent_arr = np.append(exponent_arr, exponent)
        print(exponent_arr[-1])

    exponent_ell_arr = np.array([])
    for n in n_range:
        p = 1 - a.probability(n - ell, ell=ell)
        exponent_ell = -(1 / (n - ell)) * np.log(p)
        exponent_ell_arr = np.append(exponent_ell_arr, exponent_ell)
        print(exponent_ell_arr[-1])
    fig = plt.figure()
    plt.plot(n_range, exponent_arr, n_range, exponent_ell_arr, n_range, emp_exponent)
    plt.axhline(y=0.0551, color='r', linestyle='--')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0] - 1, 120])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D_j \leq $#$D$'$_j$}$_n)$",
                r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D_j + \ell \leq $#$D$'$_j$}$_{n-\ell}}$)",
                r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
                '$D_{KL}(\Pi || Q)$'
                ])
    fig.savefig(os.path.join(args.dest_dir, 'figue1.png'))
    plt.show()











