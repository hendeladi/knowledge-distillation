import os.path
import numpy as np
import matplotlib.pyplot as plt
from src.sets import Region, SecondTerm
from simulations.utils import save_fig, save_data


def gen_exponent_plot(results_path, save_path=None, axhline=None, title=None, xlabel=None, ylabel=None, xlim=None,
                      ylim=None, legend=None):
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    exponent_arr = np.load(os.path.join(results_path, 'exponent_arr.npy'))
    exponent_ell_arr = np.load(os.path.join(results_path, 'exponent_ell_arr.npy'))
    main_term_arr = np.load(os.path.join(results_path, 'main_term_arr.npy'))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)
    fig = plt.figure()
    plt.plot(n_range, exponent_arr, n_range, exponent_ell_arr, n_range, emp_exponent)

    if axhline is not None:
        plt.axhline(y=axhline, color='r', linestyle='--')
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if legend is not None:
        plt.legend(legend)
    if save_path is not None:
        pass #fig.savefig(save_path)
    plt.show()


def example1_exponent_kd_plot():
    results_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\data\example1\exact_and_empirical\kd'
    save_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\Thesis\example1\example1_kd.png'
    axhline = 0.0551
    title = 'KD student error exponent'
    xlabel = 'n'
    ylabel = 'exponent'
    xlim = [4,120]
    ylim = [0, 0.8]
    legend = [
        r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^t_j \leq $#$D$'$^t_j$}$_n)$",
        r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D^t_j + \ell \leq $#$D$'$^t_j$}$_{n-\ell}}$)",
        r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^t_n \notin A^t_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
        '$D_{KL}(\Pi^t || Q^t)$'
    ]
    gen_exponent_plot(results_path, save_path=save_path, axhline=axhline, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim,
                      ylim=ylim, legend=legend)


def example1_exponent_baseline_plot():
    results_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\data\example1\exact_and_empirical\baseline'
    save_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\Thesis\example1\example1_baseline.png'
    axhline = 0.0173
    title = 'Baseline student error exponent'
    xlabel = 'n'
    ylabel = 'exponent'
    xlim = [4, 120]
    ylim = [0, 0.8]
    legend = [
        r"$-\frac{1}{n}lnPr(\cap_{j=1}^K ${#$D^g_j \leq $#$D$'$^g_j$}$_n)$",
        r"$-\frac{1}{n-\ell}lnPr(\cap_{j=1}^K ${#$D^g_j + \ell \leq $#$D$'$^g_j$}$_{n-\ell}}$)",
        r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A^g_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
        '$D_{KL}(\Pi^g || Q^g)$'
    ]
    gen_exponent_plot(results_path, save_path=save_path, axhline=axhline, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim,
                      ylim=ylim, legend=legend)







def ISIT2024_exponent_plot(save_flag=False):
    results_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\data\example1\exact_and_empirical\kd'
    save_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\seminar\example1_kd.png'
    axhline = 0.0551
    xlabel = 'n'
    ylabel = 'exponent'
    xlim = [4, 120]
    ylim = [0, 0.8]
    legend = [
        r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
        '$D_{KL}(\Pi || Q)$'
    ]
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    main_term_arr = np.load(os.path.join(results_path, 'main_term_arr.npy'))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)
    fig = plt.figure()
    plt.plot(n_range, emp_exponent)

    plt.axhline(y=axhline, color='r', linestyle='--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(legend)

    if save_flag:
        fig.savefig(save_path)
    plt.show()




def example1_exponent_kd_plot_seminar(save_flag=False):
    results_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\data\example1\exact_and_empirical\kd'
    save_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\seminar\example1_kd.png'
    axhline = 0.0551
    title = 'KD student error exponent'
    xlabel = 'n'
    ylabel = 'exponent'
    xlim = [4, 120]
    ylim = [0, 0.8]
    legend = [
        r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^t_n \notin A^t_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
        '$D_{KL}(\Pi^t || Q^t)$'
    ]
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    main_term_arr = np.load(os.path.join(results_path, 'main_term_arr.npy'))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)
    fig = plt.figure()
    plt.plot(n_range, emp_exponent)

    plt.axhline(y=axhline, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(legend)

    if save_flag:
        fig.savefig(save_path)
    plt.show()



def example1_exponent_baseline_plot_seminar(save_flag):
    results_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\data\example1\exact_and_empirical\baseline'
    save_path = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\seminar\example1_baseline.png'
    axhline = 0.0173
    title = 'Baseline student error exponent'
    xlabel = 'n'
    ylabel = 'exponent'
    xlim = [4, 120]
    ylim = [0, 0.7]
    legend = [
        r'$-\frac{1}{n}\ln\Pr\left(\hat{\theta}^g_n \notin A^g_{\theta_0} \mid R_{f_{opt}}(\hat{\theta}^{f_{opt}}_n)<\delta\right)$',
        '$D_{KL}(\Pi^g || Q^g)$'
    ]
    n_range = np.load(os.path.join(results_path, 'n_range.npy'))
    main_term_arr = np.load(os.path.join(results_path, 'main_term_arr.npy'))
    emp_exponent = []
    for i, n in enumerate(n_range):
        p = main_term_arr[i]
        exponent = -(1 / n) * np.log(p)
        emp_exponent.append(exponent)
    fig = plt.figure()
    plt.plot(n_range, emp_exponent)

    plt.axhline(y=axhline, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(legend)

    if save_flag:
        fig.savefig(save_path)
    plt.show()

def example1_delta_far_prob_plot_seminar(results_path):
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
    #save_fig(fig, dest_dir, "delta_far_prob.png")
    plt.show()


if __name__ == '__main__':
    print('asd')
    #example1_exponent_kd_plot()
    #example1_exponent_kd_plot_seminar(save_flag=True)
    ISIT2024_exponent_plot(save_flag=False)