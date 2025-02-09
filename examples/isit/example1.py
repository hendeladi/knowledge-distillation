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
    gt = BinaryFunction([0.3, 0.4, 0.6, 0.9])
    teacher = BinaryFunction([0.6, 0.9])
    conf = SimConfig(
        gt_func=gt,
        teacher_func=teacher,
        student_num_params=1,
        num_train_examples=list(range(2, 51, 2)),
        num_repeat=30000,
        delta=0.1,
        Aopt={"gt": Region([(0.4, 0.9)]), "kd": Region([(0, 0.9)])},
        tag="",
        dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\risk_sim_test'
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
    #plt.title(r"$Pr(R_{f_{opt}}(\hat{\theta}_n) > \delta)$ as function of number of training examples")
    plt.legend(["baseline student", "KD student"])
    plt.xlabel("number of training examples")
    plt.ylabel("probability")
    dest_dir = results_path
    save_fig(fig, dest_dir, "delta_far_prob.png")
    plt.show()

def teacher_exponent():
    D = [Region([(0.6, 0.9)])]
    Dp = [Region([(0.9, 1)])]
    ell = 2
    a = SecondTerm(D, Dp)

    n_range = list(range(4, 505, 10))
    exponent_arr = np.array([])
    for n in n_range:
        p = 1 - a.probability(n)
        exponent = -(1/n)*np.log(p)
        exponent_arr = np.append(exponent_arr, exponent)
        print(exponent_arr[-1])

    exponent_ell_arr = np.array([])
    for n in n_range:
        p = 1 - a.probability(n-ell, ell=ell)
        exponent_ell = -(1/(n-ell))*np.log(p)
        exponent_ell_arr = np.append(exponent_ell_arr, exponent_ell)
        print(exponent_ell_arr[-1])
    fig = plt.figure()
    plt.plot(n_range, exponent_arr, n_range, exponent_ell_arr)
    plt.axhline(y=0.0551, color='r', linestyle='-')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0]-1, n_range[-1]+1])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^t_j \leq $#$D$'$^t_j$}$_n)$",
                r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D^t_j + \ell \leq $#$D$'$^t_j$}$_{n-\ell}}$)",
                '$D_{KL}(\Pi^t || Q^t)$'])
    plt.title('Upper and lower bounds on excess error exponent')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\teacher_exponent'
    save_fig(fig, dest_dir, "teacher_exponent.png")
    save_data(exponent_arr, dir=dest_dir, fname="exponent_arr")
    save_data(exponent_ell_arr, dir=dest_dir, fname="exponent_ell_arr")
    save_data(n_range, dir=dest_dir, fname="n_range")
    plt.show()

def regenereate_kd_exponent(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    exponent_arr = np.load(os.path.join(results_path, 'exponent_arr.npy'))
    exponent_ell_arr = np.load(os.path.join(results_path, 'exponent_ell_arr.npy'))
    fig = plt.figure()
    plt.plot(n_range, exponent_arr, n_range, exponent_ell_arr)
    plt.axhline(y=0.0551, color='r', linestyle='--')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0] - 1, n_range[-1] + 1])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D_j \leq $#$D$'$_j$}$_n)$",
                r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D_j + \ell \leq $#$D$'$_j$}$_{n-\ell}}$)",
                '$D_{KL}(\Pi || Q)$'])
    plt.title('Upper and lower bounds on excess error exponent')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ISIT\example_1\final'
    #save_fig(fig, dest_dir, "error_exponent.png")
    plt.show()

def regenereate_kd_exponent2(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    exponent_arr = np.load(os.path.join(results_path, 'exponent_arr.npy'))
    exponent_ell_arr = np.load(os.path.join(results_path, 'exponent_ell_arr.npy'))
    main_term_arr = np.load(os.path.join(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ICML\example_1\risk_sim', 'main_term_kd_arr.npy'))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)


    fig = plt.figure()
    plt.plot(n_range, exponent_arr, n_range, exponent_ell_arr, n_range, emp_exponent)
    plt.axhline(y=0.0551, color='r', linestyle='--')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0] - 1, n_range[-1] + 1])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D_j \leq $#$D$'$_j$}$_n)$",
                r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D_j + \ell \leq $#$D$'$_j$}$_{n-\ell}}$)",
                r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
                '$D_{KL}(\Pi || Q)$'
                ])
    #plt.title('Error Exponent of excess error')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ISIT\example_1\final'
    save_fig(fig, dest_dir, "emp_error_exponent.png")
    plt.show()

def regenereate_kd_exponent3(results_path):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    #exponent_arr = np.load(os.path.join(results_path, 'exponent_arr.npy'))
    #exponent_ell_arr = np.load(os.path.join(results_path, 'exponent_ell_arr.npy'))
    main_term_arr = np.load(os.path.join(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ICML\example_1\risk_sim', 'main_term_kd_arr.npy'))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)


    fig = plt.figure()
    plt.plot(n_range, emp_exponent)
    plt.axhline(y=0.0551, color='r', linestyle='--')
    plt.ylim([0, 0.5])
    plt.xlim([n_range[0] - 1, n_range[-1] + 1])
    plt.xlabel('n')
    plt.ylabel('exponent')
    plt.legend([r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D_j \leq $#$D$'$_j$}$_n)$",
                '$D_{KL}(\Pi || Q)$'
                ])
    #plt.title('Error Exponent of excess error')
    dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ISIT2024'
    save_fig(fig, dest_dir, "emp_error_exponent.png")
    plt.show()


if __name__=='__main__':
    #teacher_exponent()
    #risk_simulation()
    #regenereate_gt_exponent(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\gt_exponent')
    #regenereate_risk(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\risk_sim')
    #regenereate_kd_exponent(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\final\example1\teacher_exponent')
    regenereate_kd_exponent3(r'C:\Users\AHENDEL\Downloads\isit2')
