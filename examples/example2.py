import os.path

import numpy as np
import matplotlib.pyplot as plt
from src.sets import Region, SecondTerm
from simulations.st_simulation import Simulation
from simulations.configs import SimConfig
from src.sets import Region
from src.function_models import BinaryFunction
from simulations.utils import save_fig, save_data


def risk_simulation():
    gt = BinaryFunction([0.5, 0.8, 0.9, 0.95])
    teacher = BinaryFunction([0.5, 0.8])
    conf = SimConfig(
        gt_func=gt,
        teacher_func=teacher,
        student_num_params=1,
        num_train_examples=list(range(4, 91, 2)),
        num_repeat=300000,
        delta=0.05,
        tag="",
        dest_dir=r'/local/mnt/workspace/ahendel/test'
    )
    sim = Simulation(conf, multi_proc=True, log=False)
    print(sim.sim_config)
    sim.run()



def regenereate_risk(results_path):
    x_axis = np.load(os.path.join(results_path, 'n_range.npy'))
    baseline_risk = np.load(os.path.join(results_path, 'delta_far_prob_baseline_arr.npy'))
    kd_risk = np.load(os.path.join(results_path, 'delta_far_prob_kd_arr.npy'))

    fig = plt.figure()
    plt.plot(x_axis, baseline_risk, 'b', x_axis, kd_risk, 'r')
    plt.title(r"$Pr(R_{f_{opt}}(\hat{\theta}_n) > \delta)$ as function of number of training examples")
    plt.legend(["baseline student", "KD student"])
    plt.xlabel("number of training examples")
    plt.ylabel("probability")
    dest_dir = results_path
    #save_fig(fig, dest_dir, "delta_far_prob.png")
    plt.show()

def regenereate_gt_exponent(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    exponent_arr = np.load(os.path.join(results_path, 'exponent_arr.npy'))
    exponent_ell_arr = np.load(os.path.join(results_path, 'exponent_ell_arr.npy'))
    fig = plt.figure()
    plt.plot(n_range, exponent_arr, n_range, exponent_ell_arr)
    plt.axhline(y=0.0173, color='r', linestyle='-')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0], n_range[-1]])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^g_j \leq $#$D$'#$^g_j$}$_n)$",
                r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D^g_j + \ell \leq $#$D$'#$^g_j$}$_{n-\ell}}$)",
                '$D_{KL}(\Pi^g || Q^g)$'])
    plt.title('Baseline student excess error exponent')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\gt_exponent'
    save_fig(fig, dest_dir, "gt_exponent.png")
    plt.show()

def regenereate_exponent_ISIT2024(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    main_term_arr = np.load(os.path.join(results_path, 'main_term_arr.npy'))
    kd_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        kd_exponent.append(exponent)

    fig = plt.figure()
    plt.plot(n_range, kd_exponent)
    plt.axhline(y=0.0551, color='r', linestyle='-')
    plt.ylim([0, 0.8])
    plt.xlim([n_range[0], n_range[-1]])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D_j \leq $#$D$'$_j$}$_n)$",
                '$D_{KL}(\Pi || Q)$'])
    #plt.title('KD student excess error exponent')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\teacher_exponent'
    # save_fig(fig, dest_dir, "gt_exponent.png")
    plt.show()

def regenereate_gt_exponent2(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    exponent_arr = np.load(os.path.join(results_path, 'baseline_exponent_arr.npy'))
    main_term_arr = np.load(os.path.join(results_path, 'main_term_baseline_arr.npy'))
    baseline_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        baseline_exponent.append(exponent)

    fig = plt.figure()
    plt.plot(n_range, baseline_exponent, n_range, exponent_arr)
    plt.axhline(y=0.0173, color='r', linestyle='-')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0], n_range[-1]])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend(['empirical'
                r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^g_j \leq $#$D$'#$^g_j$}$_n)$",
                '$D_{KL}(\Pi^g || Q^g)$'])
    plt.title('Baseline student excess error exponent')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\gt_exponent'
    #save_fig(fig, dest_dir, "gt_exponent.png")
    plt.show()

def regenereate_kd_exponent2(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    exponent_arr = np.load(os.path.join(results_path, 'kd_exponent_arr.npy'))
    exponent_arr = exponent_arr[:len(n_range)]
    main_term_arr = np.load(os.path.join(results_path, 'main_term_kd_arr.npy'))
    kd_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        kd_exponent.append(exponent)

    fig = plt.figure()
    plt.plot(n_range, kd_exponent, n_range, exponent_arr)
    plt.axhline(y=0.0551, color='r', linestyle='-')
    plt.ylim([0, 0.8])
    plt.xlim([n_range[0], n_range[-1]])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend(['empirical'
                r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^g_j \leq $#$D$'#$^g_j$}$_n)$",
                '$D_{KL}(\Pi^t || Q^t)$'])
    plt.title('KD student excess error exponent')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\teacher_exponent'
    #save_fig(fig, dest_dir, "gt_exponent.png")
    plt.show()


if __name__=='__main__':

    #risk_simulation()
    #regenereate_gt_exponent(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\gt_exponent')
    regenereate_exponent_ISIT2024(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\data\example1\exact_and_empirical\kd')
    #regenereate_risk(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ICML\example_1\risk_sim')

