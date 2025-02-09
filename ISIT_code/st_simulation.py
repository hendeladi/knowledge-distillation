import numpy as np
import random
from function_models import BinaryFunction
import os
from sets import SecondTerm
import multiprocessing
import time




def save_fig(fig=None, dir=os.getcwd(), fname="test.png"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    dest = os.path.join(dir, fname)
    fig.savefig(dest)


def save_data(arr, dir=os.getcwd(), fname="test"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    dest = os.path.join(dir, fname)
    np.save(dest, arr)


def calc_error_exp(D, Dp, n_range):
    a = SecondTerm(D, Dp)
    exponent_arr = np.array([])
    for n in n_range:
        p = 1 - a.probability(n)
        exponent = -(1/n)*np.log(p)
        exponent_arr = np.append(exponent_arr, exponent)
        #print(exponent_arr[-1])
    return(exponent_arr)


class Simulation:
    def __init__(self, sim_config, multi_proc=1):
        self.sim_config = sim_config
        self.multi_proc = multi_proc
        self.delta = sim_config.delta
        self.Aopt = sim_config.Aopt
        self.baseline_metrics = self.create_metric_dict()
        [self.opt_risk, self.opt_params, is_unique] = sim_config.gt_func.get_approx_hypoth(sim_config.student_num_params)
        self.opt_hypoth = BinaryFunction(self.opt_params)
        print('')
        self.metrics = ["risk"]
        if self.delta is not None:
            self.metrics += ["delta_far_prob"]
            if self.Aopt is not None:
                self.metrics += ["main_term"]


    @staticmethod
    def create_metric_dict():
        metric_dict = {
            "risk": [],
            "delta_far_prob": [],
            "main_term": []
        }
        return metric_dict

    def run_repeat(self, repeatitions, return_results=None):
        baseline_metrics = self.create_metric_dict()

        for n in self.sim_config.num_train_examples:
            baseline_metrics_n = self.create_metric_dict()
            print(n)

            for epoch in range(repeatitions):
                student_set = np.sort(np.array([random.random() for i in range(n)]))

                ###########  realizable ########################################
                loss_realizable, b_realizable = self.sim_config.gt_func.get_empirical_risk(self.opt_hypoth,
                                                                                       student_set,
                                                                                       self.sim_config.student_num_params)
                student_realizable = BinaryFunction(b_realizable)

                ###########  baseline ########################################
                # gt_labels = sim.gt_func.get_labels(student_set)
                loss_baseline, b_baseline = self.sim_config.gt_func.get_empirical_risk(self.sim_config.gt_func, student_set,
                                                                                       self.sim_config.student_num_params)
                student_baseline = BinaryFunction(b_baseline)


                ###########  losses ########################################
                R_baseline = BinaryFunction.get_risk(self.sim_config.gt_func, student_baseline)
                baseline_metrics_n['risk'].append(R_baseline)


                if self.delta is not None:
                    R_fopt_realizable = BinaryFunction.get_risk(self.opt_hypoth, student_realizable)
                    R_fopt_baseline = BinaryFunction.get_risk(self.opt_hypoth, student_baseline)
                    baseline_metrics_n['delta_far_prob'].append(1 if R_fopt_baseline > self.delta else 0)
                    if self.Aopt is not None:
                        if R_fopt_realizable < self.delta:
                            baseline_metrics_n['main_term'].append(1 if (not self.Aopt["gt"].contain(b_baseline)) else 0)



            ########### average over epochs ########################################
            baseline_metrics["risk"].append(np.mean(baseline_metrics_n['risk']))


            if self.delta is not None:
                baseline_metrics["delta_far_prob"].append(np.mean(baseline_metrics_n['delta_far_prob']))

                if self. Aopt is not None:
                    baseline_metrics["main_term"].append(np.mean(baseline_metrics_n['main_term']))


        if self.multi_proc>1:
            return_results.append([baseline_metrics])
        else:
            return baseline_metrics

    def run(self):
        start = time.time()
        if self.multi_proc>1:
            manager = multiprocessing.Manager()
            return_results = manager.list()
            num_processes = self.multi_proc
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
                for i in range(num_processes):
                    baseline_metric.append(return_results[i][0][metric])  # baseline
                self.baseline_metrics[metric] = np.mean(baseline_metric, axis=0)

        else:
            baseline_metrics = self.run_repeat(self.sim_config.num_repeat)
            self.baseline_metrics = baseline_metrics

        end = time.time()
        sim_time = end-start
        print(f"sim time is {sim_time//60} minutes and {sim_time%60} seconds")
        x_axis = self.sim_config.num_train_examples
        return x_axis, self.baseline_metrics

