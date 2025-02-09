import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sets import Region
from st_simulation import Simulation, calc_error_exp
from configs import CONFIGS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config to use from config dict')
    parser.add_argument('--dest_dir', type=str, default='./', help='path to save results')
    parser.add_argument('--n_cores', type=int, default=1, help='number of cpu cores to use')
    args = parser.parse_args()

    conf = CONFIGS[args.config]
    sim = Simulation(conf, multi_proc=args.n_cores)
    x_axis, kd_metrics, baseline_metrics = sim.run()
    kd_main_term = kd_metrics['main_term']
    baseline_main_term = baseline_metrics['main_term']

    if args.config == 'example1':
        D = [Region([(0.6, 0.9)])]
        Dp = [Region([(0.9, 1)])]
        kd_range = list(range(4, 121, 2))
        kd_upper_exponent = calc_error_exp(D, Dp, kd_range)

        kd_exponent = []
        for i, n in enumerate(kd_range):
            p = kd_main_term[i]
            exponent = -(1 / n) * np.log(p)
            kd_exponent.append(exponent)

        fig1 = plt.figure()
        plt.plot(kd_range, kd_upper_exponent, kd_range, kd_exponent)
        plt.axhline(y=0.0551, color='r', linestyle='-')
        plt.ylim([0, 0.8])
        plt.xlabel('n')
        plt.ylabel('exponent')
        plt.legend([r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^t_n \notin A^t_{\theta_0}\right)$',
                    r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^t_n \notin A^t_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
                    '$D_{KL}(\Pi^t || Q^t)$'])
        fig1.savefig(os.path.join(args.dest_dir, 'figue1.png'))




        D = [Region([(0.6, 0.9)]), Region([(0.4, 0.6)])]
        Dp = [Region([(0.9, 1)]), Region([(0.3, 0.4)])]
        baseline_range = list(range(4, 161, 2))
        baseline_upper_exponent = calc_error_exp(D, Dp, baseline_range)

        baseline_exponent = []
        for i, n in enumerate(baseline_range):
            p = baseline_main_term[i]
            exponent = -(1 / n) * np.log(p)
            baseline_exponent.append(exponent)
        fig2 = plt.figure()
        plt.plot(baseline_range, baseline_upper_exponent, baseline_range, baseline_exponent)
        plt.axhline(y=0.0173, color='r', linestyle='-')
        plt.ylim([0, 0.5])
        plt.xlabel('n')
        plt.ylabel('exponent')
        plt.legend([r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A^g_{\theta_0}\right)$',
                    r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A^g_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
                    '$D_{KL}(\Pi^g || Q^g)$'])

        fig2.savefig(os.path.join(args.dest_dir, 'figue2.png'))

        baseline_risk = baseline_metrics['delta_far_prob']
        kd_risk = kd_metrics['delta_far_prob']
        x_range = list(range(2, 81, 2))
        fig3 = plt.figure()
        plt.plot(x_axis, baseline_risk, 'b', x_axis, kd_risk, 'r')
        plt.legend(["baseline student", "KD student"])
        plt.xlabel("number of training examples")
        plt.ylabel("probability")

        fig3.savefig(os.path.join(args.dest_dir, 'figue3.png'))




    elif args.config == 'example2':
        baseline_risk = baseline_metrics['delta_far_prob']
        kd_risk = kd_metrics['delta_far_prob']
        x_range = list(range(2, 91, 2))
        fig3 = plt.figure()
        plt.plot(x_axis, baseline_risk, 'b', x_axis, kd_risk, 'r')
        plt.legend(["baseline student", "KD student"])
        plt.xlabel("number of training examples")
        plt.ylabel("probability")

        fig3.savefig(os.path.join(args.dest_dir, 'figue4.png'))
    else:
        pass


