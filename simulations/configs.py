from src.optimizers import optimizer_1b_subopt, optimizer_1b_sticky,optimizer_1b_min
from src.sets import Region
from src.function_models import BinaryFunction, func_1b, func_2b
import numpy as np

AllPlots = ["risk", "risk_std", "emp_risk", "gen_error", "parameters","delta_far_prob", "delta_far_prob_term1",  "delta_far_prob_term2"]
DefaultPlots = ["risk", "risk_std", "parameters", "delta_far_prob", "delta_far_prob_term1",  "delta_far_prob_term2"]


class ConfigDict:
    def __init__(self, conf_lst):
        self.config_dict = {}
        for conf in conf_lst:
            self.config_dict[conf.tag] = conf

    def __getitem__(self, item):
        return self.config_dict[item]


class SimConfig:
    def __init__(self,
                 gt_func=None,
                 teacher_func=None,
                 student_num_params=1,
                 num_train_examples=2,
                 num_repeat=1,
                 tag="Default",
                 plots=DefaultPlots,
                 dest_dir=None,
                 delta=None,
                 Aopt=None
                 ):
        self.tag = tag
        self.num_train_examples = num_train_examples
        self.num_repeat = num_repeat
        self.gt_func = gt_func
        self.teacher_func = teacher_func
        self.student_num_params = student_num_params
        self.plots = plots
        self.dest_dir = dest_dir
        self.delta = delta
        self.Aopt = Aopt

    def load_config(self, sim_config):
        self.tag = sim_config.tag
        self.num_train_examples = sim_config.num_train_examples,
        self.num_repeat = sim_config.num_repeat
        self.gt_func = sim_config.gt_func
        self.teacher_func = sim_config.teacher_func
        self.student_num_params = sim_config.student_num_params
        self.dest_dir = sim_config.dest_dir
        self.delta = sim_config.delta
        self.Aopt = sim_config.Aopt

    def __str__(self):
        lst = []
        for attr, val in self.__dict__.items():
            if isinstance(val, list):
                lst.append(f"{attr} = {[v.__str__() for v in val]}")
            else:
                lst.append(f"{attr} = {val}")

        return "\n".join(lst)




'''
############## Example 1 #############################################
def example1_gt(x):
    b1 = 0.25
    b2 = 0.5
    b3 = 0.6
    b4 = 0.9
    out = [0 if i <= b1 else 1 if i <= b2 else 0 if i <= b3 else 1 if i <= b4 else 0 for i in x]
    return np.array(out)


def example1_risk(b):
    if b <= 0.25:
        loss = 0.25-b + 0.2
    elif b <= 0.5:
        loss = b-0.25 + 0.2
    elif b <= 0.6:
        loss = 0.25 + 0.6-b
    elif b <= 0.9:
        loss = 0.25 + b-0.6
    else:
        loss = 0.25 + 0.3 + 1-b
    return loss


def example1_teacher(x):
    b1 = 0.25
    b2 = 0.9
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return out


example1_config = SimConfig(
    gt_func=example1_gt,
    gt_risk=example1_risk,
    student_func=func_1b,
    student_optimizer=optimizer_1b_subopt,
    teacher_func=example1_teacher,
    num_train_examples=120,
    num_repeat=20000,
    tag="Example 1"
)

############## Example 2 #############################################
def example2_gt(x):
    b1 = 0.25
    b2 = 0.8
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return np.array(out)


def example2_risk(b):
    if b <= 0.25:
        loss = 0.25-b + 0.2
    elif b <= 0.8:
        loss = b-0.25 + 0.2
    else:
        loss = 0.55 + 1-b
    return loss


def example2_teacher(x):
    b1 = 0.25
    out = [0 if i < b1 else 1 for i in x]
    return np.array(out)


example2_config = SimConfig(
    gt_func=example2_gt,
    gt_risk=example2_risk,
    student_func=func_1b,
    student_optimizer=optimizer_1b_sticky,
    teacher_func=example2_teacher,
    num_train_examples=100,
    num_repeat=20000,
    tag="Example 2"
)
'''
############## Example 2 - new #############################################
def example2_gt(x):
    b1 = 0.25
    b2 = 0.8
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return np.array(out)


def example2_risk(b):
    if b <= 0.25:
        loss = 0.25-b + 0.2
    elif b <= 0.8:
        loss = b-0.25 + 0.2
    else:
        loss = 0.55 + 1-b
    return loss


def example2_teacher(x):
    b1 = 0.25
    out = [0 if i < b1 else 1 for i in x]
    return np.array(out)


example2_new_config = SimConfig(
    gt_func=BinaryFunction([0.25, 0.8]),
    teacher_func=BinaryFunction([0.25]),
    student_num_params=1,
    num_train_examples=100,
    num_repeat=10000,
    tag="Example 2 - new"
)



############## Example  10 params gt #############################################
bin_func = BinaryFunction(10)
example_10params_gt = SimConfig(
    gt_func=bin_func,
    teacher_func=BinaryFunction(bin_func.get_approx_hypoth(1)[1]),
    student_num_params=1,
    num_train_examples=100,
    num_repeat=1,
    tag="example_10params_gt",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)

############## Example  2 params student #############################################
bin_func = BinaryFunction([0.1, 0.15, 0.3, 0.55, 0.6, 0.7, 0.9])
example_2params_student = SimConfig(
    gt_func=bin_func,
    teacher_func=BinaryFunction(bin_func.get_approx_hypoth(2)[1]),
    student_num_params=2,
    num_train_examples=70,
    num_repeat=10000,
    tag="example_student2params_gt7params",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)

############## Example  3 params student #############################################
bin_func = BinaryFunction([0.1, 0.15, 0.19, 0.3, 0.55, 0.6, 0.66, 0.7, 0.81, 0.93])
example_student3params_gt10params = SimConfig(
    gt_func=bin_func,
    teacher_func=BinaryFunction(bin_func.get_approx_hypoth(3)[1]),
    student_num_params=3,
    num_train_examples=70,
    num_repeat=10000,
    tag="example_student3params_gt10params",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)


############## Theory check1 #############################################
g1 = BinaryFunction([0.25, 0.3, 0.55, 0.6, 0.7])
g2 = BinaryFunction([0.05, 0.2, 0.7])
fopt = BinaryFunction([0.7])
example_theory_check1 = SimConfig(
    gt_func=fopt,
    teacher_func=[g1, g2],
    student_num_params=1,
    num_train_examples=80,
    num_repeat=20000,
    tag="example_theory_check1",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)

############## Theory check2 #############################################
g1 = BinaryFunction([0.05, 0.15, 0.7])
g2 = BinaryFunction([0.25, 0.35, 0.7])
g3 = BinaryFunction([0.45, 0.55, 0.7])
fopt = BinaryFunction([0.7])
example_theory_check2 = SimConfig(
    gt_func=fopt,
    teacher_func=[g1, g2, g3],
    student_num_params=1,
    num_train_examples=80,
    num_repeat=20000,
    tag="example_theory_check2",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)

############## Theory check3 #############################################
gt = BinaryFunction([0.45, 0.55, 0.7])
g1 = BinaryFunction([0.7])
g2 = BinaryFunction([0.45, 0.55, 0.7])
g3 = BinaryFunction([0.25, 0.35, 0.7])
example_theory_check3 = SimConfig(
    gt_func=gt,
    teacher_func=[g1, g2, g3],
    student_num_params=1,
    num_train_examples=85,
    num_repeat=20000,
    tag="example_theory_check3",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)


############## Theory check4 #############################################
g1 = BinaryFunction([0.05, 0.2, 0.7])
g2 = BinaryFunction([0.25, 0.4, 0.7])
example_theory_check4 = SimConfig(
    gt_func=g2,
    teacher_func=g1,
    student_num_params=1,
    num_train_examples=85,
    num_repeat=20000,
    tag="example_theory_check4",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)


############## Theory check5 #############################################
g1 = BinaryFunction([0.2, 0.4, 0.7])
g2 = BinaryFunction([0.2, 0.4, 0.7, 0.9, 0.95])
example_theory_check5 = SimConfig(
    gt_func=g2,
    teacher_func=g1,
    student_num_params=1,
    num_train_examples=85,
    num_repeat=20000,
    tag="example_theory_check5",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)

############## Theory check6 #############################################
g1 = BinaryFunction([0.2, 0.4, 0.7])
g2 = BinaryFunction([0.2, 0.4, 0.7, 0.9, 0.95])
example_theory_check6 = SimConfig(
    gt_func=g2,
    teacher_func=g1,
    student_num_params=1,
    num_train_examples=105,
    num_repeat=20000,
    tag="example_theory_check6",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)


############## Theory check7 #############################################
g1 = BinaryFunction([0.4, 0.5, 0.7])
g2 = BinaryFunction([0.05, 0.1, 0.4, 0.5, 0.7])
example_theory_check7 = SimConfig(
    gt_func=g2,
    teacher_func=g1,
    student_num_params=1,
    num_train_examples=120,
    num_repeat=20000,
    tag="example_theory_check7",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)


############## Theory check8 #############################################
g1 = BinaryFunction([0.4, 0.5, 0.7])
g2 = BinaryFunction([0.05, 0.15, 0.4, 0.5, 0.7])
example_theory_check8 = SimConfig(
    gt_func=g2,
    teacher_func=g1,
    student_num_params=1,
    num_train_examples=80,
    num_repeat=20000,
    tag="example_theory_check8",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)
############## Theory check8 new #############################################
g1 = BinaryFunction([0.4, 0.5, 0.7])
g2 = BinaryFunction([0.05, 0.15, 0.4, 0.5, 0.7])
fopt = BinaryFunction([0.7])
example_theory_check8_new = SimConfig(
    gt_func=fopt,
    teacher_func=[g1, g2],
    student_num_params=1,
    num_train_examples=80,
    num_repeat=20000,
    tag="example_theory_check8",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)

############## Theory check8 new #############################################
g1 = BinaryFunction([0.4, 0.5, 0.7])
g2 = BinaryFunction([0.05, 0.15, 0.4, 0.5, 0.7])
fopt = BinaryFunction([0.7])
test = SimConfig(
    gt_func=fopt,
    teacher_func=[g1, g2],
    student_num_params=1,
    num_train_examples=20,
    num_repeat=4,
    tag="test",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)



############## example1 #############################################
gt = BinaryFunction([0.2, 0.5, 0.65, 0.9])
teacher = BinaryFunction([0.2, 0.9])

example1 = SimConfig(
    gt_func=gt,
    teacher_func=teacher,
    student_num_params=1,
    num_train_examples=60,
    num_repeat=20000,
    delta=0.1,
    Aopt={"gt": Region([(0, 0.5)]), "kd": Region([(0, 0.9)])},
    tag="example1",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)
CONFIGS = ConfigDict([example_theory_check1, example_theory_check2,
                      example_theory_check3, test, example1])
















