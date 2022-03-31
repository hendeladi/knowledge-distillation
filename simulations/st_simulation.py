import numpy as np
import random
from matplotlib import pyplot as plt
from simulations.configs import example2_new_config, example_10params_gt, example_2params_student,example_student2params_gt10params
from src.function_models import BinaryFunction
from simulations.utils import save_fig, save_data
import os
from pathlib import Path
import multiprocessing
import time


class Simulation:
    def __init__(self, sim_config, multi_proc=False):
        self.sim_config = sim_config
        self.multi_proc = multi_proc
        self.baseline_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }
        self.kd_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }

    def _run_list(self, n_list, return_results=None):
        baseline_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }

        kd_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }

        for n in n_list:#range(2, self.sim_config.num_train_examples):
            R_baseline_avg = []
            R_kd_avg = []
            Remp_baseline_avg = []
            Remp_kd_avg = []
            b_kd_avg = []
            b_baseline_avg = []
            print(n)
            for epoch in range(self.sim_config.num_repeat):
                student_set = np.sort(np.array([random.random() for i in range(n)]))

                ###########  baseline ########################################
                #gt_labels = sim.gt_func.get_labels(student_set)
                loss_baseline, b_baseline = self.sim_config.gt_func.get_empirical_risk(self.sim_config.gt_func,
                                                                                       student_set,
                                                                                       self.sim_config.student_num_params)
                student_baseline = BinaryFunction(b_baseline)

                ###########  distillation ########################################
                #teacher_labels = self.sim_config.teacher_func.get_labels(student_set)
                loss_kd, b_kd = self.sim_config.teacher_func.get_empirical_risk(self.sim_config.teacher_func,
                                                                                student_set,
                                                                                self.sim_config.student_num_params)
                student_kd = BinaryFunction(b_kd)

                ###########  losses ########################################
                R_baseline = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_baseline)
                R_kd = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_kd)
                R_baseline_avg.append(R_baseline)
                R_kd_avg.append(R_kd)

                # Remp_kd = np.mean(np.abs(sim.student_func(b_kd, student_set) - gt_labels))
                Remp_baseline_avg.append(loss_baseline)
                Remp_kd_avg.append(loss_kd)
                b_baseline_avg.append(b_baseline)
                b_kd_avg.append(b_kd)

            ########### average over epochs ########################################
            baseline_metrics["risk"].append(np.mean(R_baseline_avg))
            baseline_metrics["risk_std"].append(np.std(R_baseline_avg))
            baseline_metrics["emp_risk"].append(np.mean(Remp_baseline_avg))
            baseline_metrics["parameters"].append(np.mean(b_baseline_avg, axis=0))
            kd_metrics["risk"].append(np.mean(R_kd_avg))
            kd_metrics["risk_std"].append(np.std(R_kd_avg))
            kd_metrics["emp_risk"].append(np.mean(Remp_kd_avg))
            kd_metrics["parameters"].append(np.mean(b_kd_avg, axis=0))
        if self.multi_proc:
            return_results.append([n_list, kd_metrics, baseline_metrics])
        else:
            return kd_metrics, baseline_metrics

    def run_repeat(self, repeatitions, return_results=None):
        baseline_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }

        kd_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }

        for n in range(2, self.sim_config.num_train_examples):
            R_baseline_avg = []
            R_kd_avg = []
            Remp_baseline_avg = []
            Remp_kd_avg = []
            b_kd_avg = []
            b_baseline_avg = []
            print(n)
            for epoch in range(repeatitions):
                student_set = np.sort(np.array([random.random() for i in range(n)]))

                ###########  baseline ########################################
                # gt_labels = sim.gt_func.get_labels(student_set)
                loss_baseline, b_baseline = self.sim_config.gt_func.get_empirical_risk(self.sim_config.gt_func,
                                                                                       student_set,
                                                                                       self.sim_config.student_num_params)
                student_baseline = BinaryFunction(b_baseline)

                ###########  distillation ########################################
                # teacher_labels = self.sim_config.teacher_func.get_labels(student_set)
                loss_kd, b_kd = self.sim_config.teacher_func.get_empirical_risk(self.sim_config.teacher_func,
                                                                                student_set,
                                                                                self.sim_config.student_num_params)
                student_kd = BinaryFunction(b_kd)

                ###########  losses ########################################
                R_baseline = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_baseline)
                R_kd = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_kd)
                R_baseline_avg.append(R_baseline)
                R_kd_avg.append(R_kd)

                # Remp_kd = np.mean(np.abs(sim.student_func(b_kd, student_set) - gt_labels))
                Remp_baseline_avg.append(loss_baseline)
                Remp_kd_avg.append(loss_kd)
                b_baseline_avg.append(b_baseline)
                b_kd_avg.append(b_kd)

            ########### average over epochs ########################################
            baseline_metrics["risk"].append(np.mean(R_baseline_avg))
            baseline_metrics["risk_std"].append(np.std(R_baseline_avg))
            baseline_metrics["emp_risk"].append(np.mean(Remp_baseline_avg))
            baseline_metrics["parameters"].append(np.mean(b_baseline_avg, axis=0))
            kd_metrics["risk"].append(np.mean(R_kd_avg))
            kd_metrics["risk_std"].append(np.std(R_kd_avg))
            kd_metrics["emp_risk"].append(np.mean(Remp_kd_avg))
            kd_metrics["parameters"].append(np.mean(b_kd_avg, axis=0))
        if self.multi_proc:
            return_results.append([kd_metrics, baseline_metrics])
        else:
            return kd_metrics, baseline_metrics


    def run(self):
        split_repeatitions = True
        start = time.time()
        if split_repeatitions:
            if self.multi_proc:
                manager = multiprocessing.Manager()
                return_results = manager.list()
                num_processes = 4
                repeats_per_core = self.sim_config.num_repeat//num_processes

                jobs = []
                for i in range(num_processes):
                    p = multiprocessing.Process(target=self.run_repeat, args=(repeats_per_core, return_results))
                    jobs.append(p)
                    p.start()

                for proc in jobs:
                    proc.join()

                for metric in ["risk", "risk_std", "emp_risk", "parameters"]:
                    baseline_metric = []
                    kd_metric = []
                    for i in range(num_processes):
                        kd_metric.append(return_results[i][0][metric])  # kd
                        baseline_metric.append(return_results[i][1][metric])  # baseline
                    self.kd_metrics[metric] = np.mean(kd_metric, axis=0)
                    self.baseline_metrics[metric] = np.mean(baseline_metric, axis=0)

            else:
                kd_metrics, baseline_metrics = self.run_repeat(self.sim_config.num_repeat)
                self.kd_metrics = kd_metrics
                self.baseline_metrics = baseline_metrics

        else:
            if self.multi_proc:
                manager = multiprocessing.Manager()
                return_results = manager.list()
                num_processes = 7
                n_splits = []
                for i in range(num_processes):
                    n_splits.append(list(range(2+i, self.sim_config.num_train_examples, num_processes)))

                jobs = []
                for i, n_split in enumerate(n_splits):
                    p = multiprocessing.Process(target=self._run_list, args=(n_split, return_results))
                    jobs.append(p)
                    p.start()

                for proc in jobs:
                    proc.join()

                ordered_results = {}
                for res in return_results:
                    n_lst = res[0]
                    i = n_lst[0]-2
                    ordered_results[i] = (res[1], res[2])

                def interleave(lists):
                    joined = [0]*sum([len(lst)for lst in lists])
                    num_lists = len(lists)
                    for i, lst in enumerate(lists):
                        joined[i::num_lists] = lst
                    return joined

                self.baseline_metrics["risk"] = interleave([ordered_results[i][1]["risk"] for i in range(num_processes)])
                self.baseline_metrics["risk_std"] = interleave([ordered_results[i][1]["risk_std"] for i in range(num_processes)])
                self.baseline_metrics["emp_risk"] = interleave([ordered_results[i][1]["emp_risk"] for i in range(num_processes)])
                self.baseline_metrics["parameters"] = interleave([ordered_results[i][1]["parameters"] for i in range(num_processes)])
                self.kd_metrics["risk"] = interleave([ordered_results[i][0]["risk"] for i in range(num_processes)])
                self.kd_metrics["risk_std"] = interleave([ordered_results[i][0]["risk_std"] for i in range(num_processes)])
                self.kd_metrics["emp_risk"] = interleave([ordered_results[i][0]["emp_risk"] for i in range(num_processes)])
                self.kd_metrics["parameters"] = interleave([ordered_results[i][0]["parameters"] for i in range(num_processes)])

            else:
                kd_metrics, baseline_metrics = self._run_list(list(range(2, self.sim_config.num_train_examples)))
                self.kd_metrics = kd_metrics
                self.baseline_metrics = baseline_metrics
        end = time.time()
        sim_time = end-start
        print(f"sim time is {sim_time//60} minutes and {sim_time%60} seconds")
        self.generate_plots()

        '''
        for n in range(2, self.sim_config.num_train_examples):
            R_baseline_avg = []
            R_kd_avg = []
            Remp_baseline_avg = []
            Remp_kd_avg = []
            b_kd_avg = []
            b_baseline_avg = []
            print(n)
            for epoch in range(self.sim_config.num_repeat):
                student_set = np.sort(np.array([random.random() for i in range(n)]))

                ###########  baseline ########################################
                #gt_labels = sim.gt_func.get_labels(student_set)
                loss_baseline, b_baseline = self.sim_config.gt_func.get_empirical_risk(self.sim_config.gt_func,
                                                                                       student_set,
                                                                                       self.sim_config.student_num_params)
                student_baseline = BinaryFunction(b_baseline)

                ###########  distillation ########################################
                #teacher_labels = self.sim_config.teacher_func.get_labels(student_set)
                loss_kd, b_kd = self.sim_config.teacher_func.get_empirical_risk(self.sim_config.teacher_func,
                                                                                student_set,
                                                                                self.sim_config.student_num_params)
                student_kd = BinaryFunction(b_kd)

                ###########  losses ########################################
                R_baseline = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_baseline)
                R_kd = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_kd)
                R_baseline_avg.append(R_baseline)
                R_kd_avg.append(R_kd)

                # Remp_kd = np.mean(np.abs(sim.student_func(b_kd, student_set) - gt_labels))
                Remp_baseline_avg.append(loss_baseline)
                Remp_kd_avg.append(loss_kd)
                b_baseline_avg.append(b_baseline)
                b_kd_avg.append(b_kd)

            ########### average over epochs ########################################
            self.baseline_metrics["risk"].append(np.mean(R_baseline_avg))
            self.baseline_metrics["risk_std"].append(np.std(R_baseline_avg))
            self.baseline_metrics["emp_risk"].append(np.mean(Remp_baseline_avg))
            self.baseline_metrics["parameters"].append(np.mean(b_baseline_avg, axis=0))
            self.kd_metrics["risk"].append(np.mean(R_kd_avg))
            self.kd_metrics["risk_std"].append(np.std(R_kd_avg))
            self.kd_metrics["emp_risk"].append(np.mean(Remp_kd_avg))
            self.kd_metrics["parameters"].append(np.mean(b_kd_avg, axis=0))
        '''




    def generate_plots(self):
        x_axis = np.array(range(2, self.sim_config.num_train_examples))
        gen_error_baseline_arr = np.abs(np.array(self.baseline_metrics["risk"]) - np.array(self.baseline_metrics["emp_risk"]))
        gen_error_kd_arr = np.abs(np.array(self.kd_metrics["risk"]) - np.array(self.kd_metrics["emp_risk"]))

        if "risk" in self.sim_config.plots:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["risk"], 'b', x_axis, self.kd_metrics["risk"], 'r')
            plt.title("Student test error as function of number of student training examples")
            plt.legend(["baseline", "KD"])
            plt.xlabel("number of training examples")
            plt.ylabel("error")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "risk.png")
                save_data(self.baseline_metrics["risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_baseline_arr")
                save_data(self.kd_metrics["risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_kd_arr")

        if "emp_risk" in self.sim_config.plots:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["emp_risk"], 'b', x_axis, self.kd_metrics["emp_risk"], 'r')
            plt.title("Student empirical error as function of number of student training examples")
            plt.legend(["baseline", "KD"])
            plt.xlabel("number of training examples")
            plt.ylabel("error")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "emp_risk.png")
                save_data(self.baseline_metrics["emp_risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "Remp_baseline_arr")
                save_data(self.kd_metrics["emp_risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "Remp_kd_arr")

        if "gen_error" in self.sim_config.plots:
            fig = plt.figure()
            plt.plot(x_axis, gen_error_baseline_arr, 'b', x_axis, gen_error_kd_arr, 'r')
            plt.title("Student generalization error as function of number of student training examples")
            plt.legend(["baseline", "KD"])
            plt.xlabel("number of training examples")
            plt.ylabel("error")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "gen_error.png")
                save_data(gen_error_baseline_arr, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "gen_error_baseline_arr")
                save_data(gen_error_kd_arr, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "gen_error_kd_arr")

        if "risk_std" in self.sim_config.plots:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["risk_std"], 'b', x_axis, self.kd_metrics["risk_std"], 'r')
            plt.title("Student std function of number of student training examples")
            plt.legend(["baseline", "KD"])
            plt.xlabel("number of training examples")
            plt.ylabel("std")

        if "parameters" in self.sim_config.plots:
            for i in range(self.sim_config.student_num_params):
                baseline_param_vals = [p[i] for p in self.baseline_metrics["parameters"]]
                kd_param_vals = [p[i] for p in self.kd_metrics["parameters"]]
                fig = plt.figure()
                plt.plot(x_axis, baseline_param_vals, 'b', x_axis, kd_param_vals, 'r')
                plt.title(f"expected parameter {i} value as function of number of training examples")
                plt.legend([r"theta_baseline", r"theta_kd"])
                plt.xlabel("number of training examples")
                plt.ylabel("parameter value")



if __name__ == '__main__':

    a = example_student2params_gt10params
    a.dest_dir = "/results/"
    a.num_repeat = 4
    sim = Simulation(a, multi_proc=True)
    sim.run()
    #plt.show()
