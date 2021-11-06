import numpy as np
import random
from matplotlib import pyplot as plt
from configs import SimConfig, example1_config, example2_config, example3_config, example4_config,example2v2_config
from utils import save_fig, save_data
import os
from pathlib import Path


example2_config_save = example2_config
example2_config_save.dest_dir = None#r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
simulations = [
    example2_config_save
]

for sim in simulations:
    print(f"""Starting simulation:\n tag = {sim.tag}\n num_train_examples = {sim.num_train_examples}\n num_repeat = {sim.num_repeat}""")
    R_baseline_arr = []
    R_kd_arr = []
    Remp_baseline_arr = []
    Remp_kd_arr = []
    b_kd_arr = []
    b_baseline_arr = []
    KD_better_prob = []
    baseline_better_prob = []
    for n in range(2, sim.num_train_examples):
        N_student = n
        R_baseline_avg = []
        R_kd_avg = []
        Remp_baseline_avg = []
        Remp_kd_avg = []
        b_kd_avg = []
        b_baseline_avg = []
        KD_better_count = 0
        baseline_better_count = 0
        print(n)
        for epoch in range(sim.num_repeat):
            student_set = np.sort(np.array([random.random() for i in range(N_student)]))

            ###########  baseline ########################################
            gt_labels = sim.gt_func(student_set)
            b_baseline, loss_baseline = sim.student_optimizer(student_set, gt_labels)

            ###########  distillation ########################################
            teacher_labels = sim.teacher_func(student_set)
            b_kd, loss_kd = sim.student_optimizer(student_set, teacher_labels)

            ###########  losses ########################################
            R_baseline = sim.gt_risk(b_baseline)
            R_kd = sim.gt_risk(b_kd)
            R_baseline_avg.append(R_baseline)
            R_kd_avg.append(R_kd)

            Remp_baseline = loss_baseline
            Remp_kd = np.mean(np.abs(sim.student_func(b_kd, student_set) - gt_labels))
            Remp_baseline_avg.append(Remp_baseline)
            Remp_kd_avg.append(Remp_kd)
            b_baseline_avg.append(b_baseline)
            b_kd_avg.append(b_kd)
            KD_better_count += int(R_kd < R_baseline)
            baseline_better_count += int(R_kd > R_baseline)
            if abs(b_kd-0.25) > abs(b_baseline-0.25):
                print("b_kd = {}, b_baseline = {}".format(b_kd, b_baseline))
        ########### average over epochs ########################################
        R_baseline_arr.append(np.mean(R_baseline_avg))
        R_kd_arr.append(np.mean(R_kd_avg))
        Remp_baseline_arr.append(np.mean(Remp_baseline_avg))
        Remp_kd_arr.append(np.mean(Remp_kd_avg))
        b_baseline_arr.append(np.mean(b_baseline_avg))
        b_kd_arr.append(np.mean(b_kd_avg))
        KD_better_prob.append(KD_better_count/sim.num_repeat)
        baseline_better_prob.append(baseline_better_count / sim.num_repeat)


    print(f"b_baseline = {b_baseline},   b_kd = {b_kd}")
    gen_error_baseline_arr = np.abs(np.array(R_baseline_arr) - np.array(Remp_baseline_arr))
    gen_error_kd_arr = np.abs(np.array(R_kd_arr) - np.array(Remp_kd_arr))

    x_axis = np.array(range(2, sim.num_train_examples))
    fig = plt.figure()
    plt.plot(x_axis, R_baseline_arr, 'b', x_axis, R_kd_arr, 'r')
    plt.title("Student test error as function of number of student training examples")
    plt.legend(["baseline", "KD"])
    plt.xlabel("number of training examples")
    plt.ylabel("error")
    if sim.dest_dir is not None:
        Path(sim.dest_dir).mkdir(parents=True, exist_ok=True)
        save_fig(fig, os.path.join(sim.dest_dir, sim.tag), "risk.png")
        save_data(R_baseline_arr, os.path.join(sim.dest_dir, sim.tag), "R_baseline_arr")
        save_data(R_kd_arr, os.path.join(sim.dest_dir, sim.tag), "R_kd_arr")

    fig = plt.figure()
    plt.plot(x_axis, Remp_baseline_arr, 'b', x_axis, Remp_kd_arr, 'r')
    plt.title("Student empirical error as function of number of student training examples")
    plt.legend(["baseline", "KD"])
    plt.xlabel("number of training examples")
    plt.ylabel("error")

    if sim.dest_dir is not None:
        Path(sim.dest_dir).mkdir(parents=True, exist_ok=True)
        save_fig(fig, os.path.join(sim.dest_dir, sim.tag), "emp_risk.png")
        save_data(Remp_baseline_arr, os.path.join(sim.dest_dir, sim.tag), "Remp_baseline_arr")
        save_data(Remp_kd_arr, os.path.join(sim.dest_dir, sim.tag), "Remp_kd_arr")
    realizable = 0.5 * np.log(x_axis) / x_axis
    non_realizable = 0.5 * 1 / np.sqrt(x_axis)
    fig = plt.figure()
    plt.plot(x_axis, gen_error_baseline_arr, 'b', x_axis, gen_error_kd_arr, 'r', x_axis, non_realizable, '--b', x_axis, realizable, '--r')
    plt.title("Student generalization error as function of number of student training examples")
    plt.legend(["baseline", "KD", 'non-realizable rate', 'realizable rate'])
    plt.xlabel("number of training examples")
    plt.ylabel("error")
    if sim.dest_dir is not None:
        Path(sim.dest_dir).mkdir(parents=True, exist_ok=True)
        save_fig(fig, os.path.join(sim.dest_dir, sim.tag), "gen_error.png")
        save_data(gen_error_baseline_arr, os.path.join(sim.dest_dir, sim.tag), "gen_error_baseline_arr")
        save_data(gen_error_kd_arr, os.path.join(sim.dest_dir, sim.tag), "gen_error_kd_arr")


    fig = plt.figure()
    plt.plot(x_axis, baseline_better_prob, 'b', x_axis, KD_better_prob, 'r')
    plt.title("probability KD risk is smaller than basesline risk as function of number of student training examples")
    plt.legend([r"Pr(R_kd > R_baseline)", r"Pr(R_kd < R_baseline)"])
    plt.xlabel("number of training examples")
    plt.ylabel("probability")
    if sim.dest_dir is not None:
        Path(sim.dest_dir).mkdir(parents=True, exist_ok=True)
        save_fig(fig, os.path.join(sim.dest_dir, sim.tag), "KD_better_prob.png")
        save_data(KD_better_prob, os.path.join(sim.dest_dir, sim.tag), "KD_better_prob")
        save_data(baseline_better_prob, os.path.join(sim.dest_dir, sim.tag), "baseline_better_prob")


    fig = plt.figure()
    plt.plot(x_axis, b_baseline_arr, 'b', x_axis, b_kd_arr, 'r')
    plt.title("expected parameter value as function of number of training examples")
    plt.legend([r"theta_baseline", r"theta_kd"])
    plt.xlabel("number of training examples")
    plt.ylabel("parameter value")
    plt.axhline(y=0.25, color='black', linestyle='--')
    if sim.dest_dir is not None:
        Path(sim.dest_dir).mkdir(parents=True, exist_ok=True)
        save_fig(fig, os.path.join(sim.dest_dir, sim.tag), "expected_parameter.png")
        save_data(b_kd_arr, os.path.join(sim.dest_dir, sim.tag), "theta_kd")
        save_data(b_baseline_arr, os.path.join(sim.dest_dir, sim.tag), "theta_baseline")

plt.show()

