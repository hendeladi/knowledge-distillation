import numpy as np
import random
from matplotlib import pyplot as plt
from configs import example2_new_config, example_10params_gt,example_2params_student
from src.function_models import BinaryFunction
from utils import save_fig, save_data
import os
from pathlib import Path
import time





simulations = [
    example_2params_student
]

for sim in simulations:
    start_time = time.time()
    print(f"""Starting simulation""")
    print(sim)
    R_baseline_arr = []
    R_kd_arr = []
    R_baseline_std = []
    R_kd_std = []
    Remp_baseline_arr = []
    Remp_kd_arr = []
    b_kd_arr = []
    b_baseline_arr = []
    for n in range(2, sim.num_train_examples):
        N_student = n
        R_baseline_avg = []
        R_kd_avg = []
        Remp_baseline_avg = []
        Remp_kd_avg = []
        b_kd_avg = []
        b_baseline_avg = []
        print(n)
        for epoch in range(sim.num_repeat):
            student_set = np.sort(np.array([random.random() for i in range(N_student)]))

            ###########  baseline ########################################
            gt_labels = sim.gt_func.get_labels(student_set)
            loss_baseline, b_baseline = sim.gt_func.get_empirical_risk(sim.gt_func, student_set, sim.student_num_params)
            student_baseline = BinaryFunction(b_baseline)

            ###########  distillation ########################################
            teacher_labels = sim.teacher_func.get_labels(student_set)
            loss_kd, b_kd = sim.teacher_func.get_empirical_risk(sim.teacher_func, student_set, sim.student_num_params)
            student_kd = BinaryFunction(b_kd)

            ###########  losses ########################################
            R_baseline = sim.gt_func.get_risk(sim.gt_func, student_baseline)
            R_kd = sim.gt_func.get_risk(sim.gt_func, student_kd)
            R_baseline_avg.append(R_baseline)
            R_kd_avg.append(R_kd)

            #Remp_baseline = loss_baseline
            #Remp_kd = np.mean(np.abs(sim.student_func(b_kd, student_set) - gt_labels))
            Remp_baseline_avg.append(loss_baseline)
            Remp_kd_avg.append(loss_kd)
            b_baseline_avg.append(b_baseline)
            b_kd_avg.append(b_kd)
            #if abs(b_kd-0.25) > abs(b_baseline-0.25):
            #    print("b_kd = {}, b_baseline = {}".format(b_kd, b_baseline))
        ########### average over epochs ########################################
        R_baseline_arr.append(np.mean(R_baseline_avg))
        R_kd_arr.append(np.mean(R_kd_avg))
        R_baseline_std.append(np.std(R_baseline_avg))
        R_kd_std.append(np.std(R_kd_avg))
        Remp_baseline_arr.append(np.mean(Remp_baseline_avg))
        Remp_kd_arr.append(np.mean(Remp_kd_avg))
        b_baseline_arr.append(np.mean(b_baseline_avg))
        b_kd_arr.append(np.mean(b_kd_avg))
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print(f"b_baseline = {b_baseline},   b_kd = {b_kd}\n")
    print(f"Sim time = {time_elapsed//60} minutes and {time_elapsed%60} seconds" )




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
    fig = plt.figure()
    plt.plot(x_axis, gen_error_baseline_arr, 'b', x_axis, gen_error_kd_arr, 'r')
    plt.title("Student generalization error as function of number of student training examples")
    plt.legend(["baseline", "KD"])
    plt.xlabel("number of training examples")
    plt.ylabel("error")
    if sim.dest_dir is not None:
        Path(sim.dest_dir).mkdir(parents=True, exist_ok=True)
        save_fig(fig, os.path.join(sim.dest_dir, sim.tag), "gen_error.png")
        save_data(gen_error_baseline_arr, os.path.join(sim.dest_dir, sim.tag), "gen_error_baseline_arr")
        save_data(gen_error_kd_arr, os.path.join(sim.dest_dir, sim.tag), "gen_error_kd_arr")

    fig = plt.figure()
    plt.plot(x_axis, R_baseline_std, 'b', x_axis, R_kd_std, 'r')
    plt.title("Student std function of number of student training examples")
    plt.legend(["baseline", "KD"])
    plt.xlabel("number of training examples")
    plt.ylabel("std")
'''
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
'''
plt.show()

