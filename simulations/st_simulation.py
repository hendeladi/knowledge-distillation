import numpy as np
import logging
import random
from matplotlib import pyplot as plt
from src.function_models import BinaryFunction
from simulations.utils import save_fig, save_data
import os
from pathlib import Path
import multiprocessing
import time
from datetime import datetime

NUM_PROCESSES = 4


class Simulation:
    def __init__(self, sim_config, multi_proc=False, log=False):
        self.sim_config = sim_config
        self.multi_proc = multi_proc
        self.log = log
        self.delta = sim_config.delta
        self.Aopt = sim_config.Aopt
        self.baseline_metrics = self.create_metric_dict()
        self.kd_metrics = self.create_metric_dict()
        [self.opt_risk, self.opt_params, is_unique] = sim_config.gt_func.get_approx_hypoth(sim_config.student_num_params)
        self.opt_hypoth = BinaryFunction(self.opt_params)
        print('')
        self.metrics = ["risk", "risk_std", "emp_risk", "parameters"]
        if self.delta is not None:
            self.metrics += ["delta_far_prob"]
            if self.Aopt is not None:
                self.metrics += ["delta_far_prob_term1", "delta_far_prob_term2"]


    @staticmethod
    def create_metric_dict():
        metric_dict = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": [],
            "delta_far_prob": [],
            "delta_far_prob_term1": [],
            "delta_far_prob_term2": []
        }
        return metric_dict


    def run_repeat(self, repeatitions, return_results=None):
        if self.log:
            logging.basicConfig(filename="Log_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")+".txt",
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
        baseline_metrics = self.create_metric_dict()

        kd_metrics = self.create_metric_dict()

        for n in range(2, self.sim_config.num_train_examples):
            baseline_metrics_n = self.create_metric_dict()
            kd_metrics_n = self.create_metric_dict()
            print(n)

            if n%2 == 0 and self.log:
                logging.info('on example {}'.format(n))
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
                R_baseline = BinaryFunction.get_risk(self.sim_config.gt_func, student_baseline)
                R_kd = BinaryFunction.get_risk(self.sim_config.gt_func, student_kd)

                baseline_metrics_n['risk'].append(R_baseline)#R_baseline_avg.append(R_baseline)
                kd_metrics_n['risk'].append(R_kd)#R_kd_avg.append(R_kd)

                # Remp_kd = np.mean(np.abs(sim.student_func(b_kd, student_set) - gt_labels))
                baseline_metrics_n['emp_risk'].append(loss_baseline)#Remp_baseline_avg.append(loss_baseline)
                kd_metrics_n['emp_risk'].append(loss_kd)#Remp_kd_avg.append(loss_kd)
                baseline_metrics_n['parameters'].append(b_baseline)#b_baseline_avg.append(b_baseline)
                kd_metrics_n['parameters'].append(b_kd)#b_kd_avg.append(b_kd)
                if self.delta is not None:
                    R_fopt_baseline = BinaryFunction.get_risk(self.opt_hypoth, student_baseline)
                    R_fopt_kd = BinaryFunction.get_risk(self.opt_hypoth, student_kd)
                    baseline_metrics_n['delta_far_prob'].append(1 if R_fopt_baseline > self.delta else 0)
                    kd_metrics_n['delta_far_prob'].append(1 if R_fopt_kd > self.delta else 0)
                    if self.Aopt is not None:
                        baseline_metrics_n['delta_far_prob_term1'].append(1 if (self.Aopt["gt"].contain(b_baseline) and R_fopt_baseline > self.delta) else 0)
                        baseline_metrics_n['delta_far_prob_term2'].append(1 if (not self.Aopt["gt"].contain(b_baseline)) else 0)
                        kd_metrics_n['delta_far_prob_term1'].append(1 if (self.Aopt["kd"].contain(b_kd) and R_fopt_kd > self.delta) else 0)
                        kd_metrics_n['delta_far_prob_term2'].append(1 if (not self.Aopt["kd"].contain(b_kd)) else 0)


            ########### average over epochs ########################################
            baseline_metrics["risk"].append(np.mean(baseline_metrics_n['risk']))#baseline_metrics["risk"].append(np.mean(R_baseline_avg))
            kd_metrics["risk"].append(np.mean(kd_metrics_n['risk']))  # kd_metrics["risk"].append(np.mean(R_kd_avg))

            baseline_metrics["risk_std"].append(np.std(baseline_metrics_n['risk']))#baseline_metrics["risk_std"].append(np.std(R_baseline_avg))
            kd_metrics["risk_std"].append(np.std(kd_metrics_n['risk']))  # kd_metrics["risk_std"].append(np.std(R_kd_avg))

            baseline_metrics["emp_risk"].append(np.mean(baseline_metrics_n['emp_risk']))#baseline_metrics["emp_risk"].append(np.mean(Remp_baseline_avg))
            kd_metrics["emp_risk"].append(np.mean(kd_metrics_n['emp_risk']))  # kd_metrics["emp_risk"].append(np.mean(Remp_kd_avg))

            baseline_metrics["parameters"].append(np.mean(baseline_metrics_n['parameters'], axis=0))#baseline_metrics["parameters"].append(np.mean(b_baseline_avg, axis=0))
            kd_metrics["parameters"].append(np.mean(kd_metrics_n['parameters'], axis=0))  # kd_metrics["parameters"].append(np.mean(b_kd_avg, axis=0))

            if self.delta is not None:
                baseline_metrics["delta_far_prob"].append(np.mean(baseline_metrics_n['delta_far_prob']))
                kd_metrics["delta_far_prob"].append(np.mean(kd_metrics_n['delta_far_prob']))
                if self. Aopt is not None:
                    baseline_metrics["delta_far_prob_term1"].append(np.mean(baseline_metrics_n['delta_far_prob_term1']))
                    kd_metrics["delta_far_prob_term1"].append(np.mean(kd_metrics_n['delta_far_prob_term1']))
                    baseline_metrics["delta_far_prob_term2"].append(np.mean(baseline_metrics_n['delta_far_prob_term2']))
                    kd_metrics["delta_far_prob_term2"].append(np.mean(kd_metrics_n['delta_far_prob_term2']))

        if self.multi_proc:
            return_results.append([kd_metrics, baseline_metrics])
        else:
            return kd_metrics, baseline_metrics

    def run(self):
        start = time.time()
        if self.multi_proc:
            manager = multiprocessing.Manager()
            return_results = manager.list()
            num_processes = NUM_PROCESSES
            repeats_per_core = self.sim_config.num_repeat//num_processes

            jobs = []
            for i in range(num_processes):
                p = multiprocessing.Process(target=self.run_repeat, args=(repeats_per_core, return_results))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()

            for metric in self.metrics:
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

        end = time.time()
        sim_time = end-start
        print(f"sim time is {sim_time//60} minutes and {sim_time%60} seconds")
        self.generate_plots()

    def generate_plots(self):
        x_axis = np.array(range(2, self.sim_config.num_train_examples))
        gen_error_baseline_arr = np.abs(np.array(self.baseline_metrics["risk"]) - np.array(self.baseline_metrics["emp_risk"]))
        gen_error_kd_arr = np.abs(np.array(self.kd_metrics["risk"]) - np.array(self.kd_metrics["emp_risk"]))

        if "risk" in self.sim_config.plots:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["risk"], 'b', x_axis, self.kd_metrics["risk"], 'r')
            plt.title("Student risk as function of number of training examples")
            plt.legend(["baseline student", "KD student"])
            plt.xlabel("number of training examples")
            plt.ylabel("risk")
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
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "risk_std.png")
                save_data(self.baseline_metrics["risk_std"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_baseline_std_arr")
                save_data(self.kd_metrics["risk_std"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_kd_std_arr")

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
                if self.sim_config.dest_dir is not None:
                    Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                    save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "params.png")

        if "delta_far_prob" in self.sim_config.plots and self.delta is not None:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["delta_far_prob"], 'b', x_axis, self.kd_metrics["delta_far_prob"], 'r')
            plt.title(f"Student risk >{self.delta} probability as function of number of training examples")
            plt.legend(["baseline student", "KD student"])
            plt.xlabel("number of training examples")
            plt.ylabel("probability")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob.png")
                save_data(self.baseline_metrics["delta_far_prob"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_baseline_arr")
                save_data(self.kd_metrics["delta_far_prob"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_kd_arr")

        if "delta_far_prob_term1" in self.sim_config.plots and self.Aopt is not None:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["delta_far_prob_term1"], 'b', x_axis, self.kd_metrics["delta_far_prob_term1"], 'r')
            plt.title(f"Student risk >{self.delta} term1 probability as function of number of training examples")
            plt.legend(["baseline student", "KD student"])
            plt.xlabel("number of training examples")
            plt.ylabel("probability")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_term1.png")
                save_data(self.baseline_metrics["delta_far_prob_term1"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_term1_baseline_arr")
                save_data(self.kd_metrics["delta_far_prob_term1"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_term1_kd_arr")

        if "delta_far_prob_term2" in self.sim_config.plots and self.Aopt is not None:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["delta_far_prob_term2"], 'b', x_axis, self.kd_metrics["delta_far_prob_term2"], 'r')
            plt.title(f"Student risk >{self.delta} term2 probability as function of number of training examples")
            plt.legend(["baseline student", "KD student"])
            plt.xlabel("number of training examples")
            plt.ylabel("probability")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_term2.png")
                save_data(self.baseline_metrics["delta_far_prob_term2"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_term2_baseline_arr")
                save_data(self.kd_metrics["delta_far_prob_term2"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "delta_far_prob_term2_kd_arr")

        if "delta_far_prob_term1" in self.sim_config.plots and "delta_far_prob_term2" in self.sim_config.plots and self.Aopt is not None:
            fig = plt.figure()
            plt.plot(x_axis, self.baseline_metrics["delta_far_prob"], 'b', x_axis, self.baseline_metrics["delta_far_prob_term1"], 'r',x_axis, self.baseline_metrics["delta_far_prob_term2"], 'g')
            plt.title(f"Baseline student delta = {self.delta} probabilities as function of number of training examples")
            plt.legend(["delta_far_prob", "term1", "term2"])
            plt.xlabel("number of training examples")
            plt.ylabel("probability")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "baseline_delta_probs.png")

            fig = plt.figure()
            plt.plot(x_axis, self.kd_metrics["delta_far_prob"], 'b', x_axis,
                     self.kd_metrics["delta_far_prob_term1"], 'r', x_axis,
                     self.kd_metrics["delta_far_prob_term2"], 'g')
            plt.title(f"KD student delta = {self.delta} probabilities as function of number of training examples")
            plt.legend(["delta_far_prob", "term1", "term2"])
            plt.xlabel("number of training examples")
            plt.ylabel("probability")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "kd_delta_probs.png")


class Simulation2:
    def __init__(self, sim_config, multi_proc=False, log=False):
        self.sim_config = sim_config
        self.multi_proc = multi_proc
        self.log = log
        self.baseline_metrics = {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        }

        self.kd_metrics = {i: {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        } for i in range(len(self.sim_config.teacher_func))}

    def run_repeat(self, repeatitions, return_results=None):
        if self.log:
            logging.basicConfig(filename="Log_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")+".txt",
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)

        kd_metrics = {i: {
            "risk": [],
            "risk_std": [],
            "emp_risk": [],
            "parameters": []
        } for i in range(len(self.sim_config.teacher_func))}

        for n in range(2, self.sim_config.num_train_examples):
            R_kd_avg = {i: [] for i in range(len(self.sim_config.teacher_func))}
            Remp_kd_avg = {i: [] for i in range(len(self.sim_config.teacher_func))}
            b_kd_avg = {i: [] for i in range(len(self.sim_config.teacher_func))}
            print(n)

            if n % 2 == 0 and self.log:
                logging.info('on example {}'.format(n))
            for epoch in range(repeatitions):
                student_set = np.sort(np.array([random.random() for i in range(n)]))


                ###########  distillation ########################################
                for i in range(len(self.sim_config.teacher_func)):
                    loss_kd, b_kd = self.sim_config.teacher_func[i].get_empirical_risk(self.sim_config.teacher_func[i],
                                                                                    student_set,
                                                                                    self.sim_config.student_num_params)
                    student_kd = BinaryFunction(b_kd)
                    R_kd = self.sim_config.gt_func.get_risk(self.sim_config.gt_func, student_kd)
                    R_kd_avg[i].append(R_kd)
                    Remp_kd_avg[i].append(loss_kd)
                    b_kd_avg[i].append(b_kd)
            for i in range(len(self.sim_config.teacher_func)):
                kd_metrics[i]["risk"].append(np.mean(R_kd_avg[i]))
                kd_metrics[i]["risk_std"].append(np.std(R_kd_avg[i]))
                kd_metrics[i]["emp_risk"].append(np.mean(Remp_kd_avg[i]))
                kd_metrics[i]["parameters"].append(np.mean(b_kd_avg[i], axis=0))
        if self.multi_proc:
            return_results.append([kd_metrics])
        else:
            return kd_metrics

    def run(self):
        start = time.time()
        if self.multi_proc:
            manager = multiprocessing.Manager()
            return_results = manager.list()
            num_processes = NUM_PROCESSES
            repeats_per_core = self.sim_config.num_repeat//num_processes

            jobs = []
            for i in range(num_processes):
                p = multiprocessing.Process(target=self.run_repeat, args=(repeats_per_core, return_results))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            for t in range(len(self.sim_config.teacher_func)):
                for metric in ["risk", "risk_std", "emp_risk", "parameters"]:
                    kd_metric = []
                    for j in range(num_processes):
                        proc_kd_metric = return_results[j][0]
                        kd_metric.append(proc_kd_metric[t][metric])
                    self.kd_metrics[t][metric] = np.mean(kd_metric, axis=0)
        else:
            self.kd_metrics = self.run_repeat(self.sim_config.num_repeat)

        end = time.time()
        sim_time = end-start
        print(f"sim time is {sim_time//60} minutes and {sim_time%60} seconds")
        self.generate_plots()

    def generate_plots(self):
        x_axis = np.array(range(2, self.sim_config.num_train_examples))
        if "risk" in self.sim_config.plots:
            fig = plt.figure()
            for t in range(len(self.sim_config.teacher_func)):
                plt.plot(x_axis, self.kd_metrics[t]["risk"])
            plt.title("Student test error as function of number of student training examples")
            plt.legend(["student {}".format(i) for i in range(len(self.sim_config.teacher_func))])
            plt.xlabel("number of training examples")
            plt.ylabel("error")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "risk.png")
                for t in range(len(self.sim_config.teacher_func)):
                    save_data(self.kd_metrics[t]["risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_kd_{}".format(t))


        if "emp_risk" in self.sim_config.plots:
            fig = plt.figure()
            for t in range(len(self.sim_config.teacher_func)):
                plt.plot(x_axis, self.kd_metrics[t]["emp_risk"])
            plt.title("Student empirical error as function of number of student training examples")
            plt.legend(["student {}".format(i) for i in range(len(self.sim_config.teacher_func))])
            plt.xlabel("number of training examples")
            plt.ylabel("error")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "emp_risk.png")
                #save_data(self.baseline_metrics["emp_risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "Remp_baseline_arr")
                #save_data(self.kd_metrics["emp_risk"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "Remp_kd_arr")


        if "risk_std" in self.sim_config.plots:
            fig = plt.figure()
            for t in range(len(self.sim_config.teacher_func)):
                plt.plot(x_axis, self.kd_metrics[t]["risk_std"])
            plt.title("Student std function of number of student training examples")
            plt.legend(["student {}".format(i) for i in range(len(self.sim_config.teacher_func))])
            plt.xlabel("number of training examples")
            plt.ylabel("std")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "risk_std.png")
                #save_data(self.baseline_metrics["risk_std"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_baseline_std_arr")
                #save_data(self.kd_metrics["risk_std"], os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "R_kd_std_arr")

        if "parameters" in self.sim_config.plots:
            fig = plt.figure()
            for t in range(len(self.sim_config.teacher_func)):
                plt.plot(x_axis, self.kd_metrics[t]["parameters"])
            plt.title(f"expected parameter value as function of number of training examples")
            plt.legend(["student {}".format(i) for i in range(len(self.sim_config.teacher_func))])
            plt.xlabel("number of training examples")
            plt.ylabel("parameter value")
            if self.sim_config.dest_dir is not None:
                Path(self.sim_config.dest_dir).mkdir(parents=True, exist_ok=True)
                save_fig(fig, os.path.join(self.sim_config.dest_dir, self.sim_config.tag), "params.png")




