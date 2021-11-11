from src.optimizers import optimizer_1b_subopt, optimizer_1b_sticky,optimizer_1b_min
from src.function_models import BinaryFunction, func_1b, func_2b
import numpy as np


class SimConfig:
    def __init__(self,
                 gt_func=None,
                 teacher_func=None,
                 student_num_params=1,
                 num_train_examples=2,
                 num_repeat=1,
                 tag="Default",
                 dest_dir=None
                 ):
        self.tag = tag
        self.num_train_examples = num_train_examples
        self.num_repeat = num_repeat
        self.gt_func = gt_func
        self.teacher_func = teacher_func
        self.student_num_params = student_num_params
        self.dest_dir = dest_dir

    def load_config(self, sim_config):
        self.tag = sim_config.tag
        self.num_train_examples = sim_config.num_train_examples,
        self.num_repeat = sim_config.num_repeat
        self.gt_func = sim_config.gt_func
        self.teacher_func = sim_config.teacher_func
        self.student_num_params = sim_config.student_num_params
        self.dest_dir = sim_config.dest_dir

    def __str__(self):
        return "\n".join([f"{attr} = {val}" for attr, val in self.__dict__.items()])


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
    num_train_examples=50,
    num_repeat=10000,
    tag="example_student2params_gt7params",
    dest_dir=r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results'
)
'''
############## Example 2 v2 #############################################

def example2v2_gt(x):
    b1 = 0.25
    b2 = 0.7
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return np.array(out)


def example2v2_risk(b):
    b1 = 0.25
    b2 = 0.7
    if b <= b1:
        loss = b1-b + 1-b2
    elif b <= b2:
        loss = b-b1 + 1-b2
    else:
        loss = (b2-b1) + 1-b
    return loss


def example2v2_teacher(x):
    b1 = 0.25
    out = [0 if i < b1 else 1 for i in x]
    return np.array(out)


example2v2_config = SimConfig(
    gt_func=example2v2_gt,
    gt_risk=example2v2_risk,
    student_func=func_1b,
    student_optimizer=optimizer_1b_subopt,
    teacher_func=example2v2_teacher,
    num_train_examples=130,
    num_repeat=20000,
    tag="Example 2 v2"
)



############## Example 3 #############################################
def example3_gt(x):
    b1 = 0.25
    b2 = 0.8
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return np.array(out)


def example3_risk(b):
    if b <= 0.25:
        loss = 0.25-b + 0.2
    elif b <= 0.8:
        loss = b-0.25 + 0.2
    else:
        loss = 0.55 + 1-b
    return loss


def example3_teacher(x):
    b1 = 0.1
    b2 = 0.15
    b3 = 0.25
    b4 = 0.8
    out = [0 if i < b1 else 1 if i < b2 else 0 if i < b3 else 1 if i < b4 else 0 for i in x]
    return out


example3_config = SimConfig(
    gt_func=example3_gt,
    gt_risk=example3_risk,
    student_func=func_1b,
    student_optimizer=optimizer_1b_subopt,
    teacher_func=example3_teacher,
    num_train_examples=120,
    num_repeat=20000,
    tag="Example 3"
)


############## Example 4 #############################################
def example4_gt(x):
    b1 = 0.1
    b2 = 0.3
    out = [0 if i <= b1 else 1 if i <= b2 else 0 for i in x]
    return out

def example4_risk(b):
    if b <= 0.1:
        loss = 0.1-b + 0.7
    elif b <= 0.3:
        loss = b-0.1 + 0.7
    else:
        loss = 0.1 + 1-b
    return loss


def example4_teacher(x):
    b1 = 0.1
    b2 = 0.4
    out = [0 if i < b1 else 1 if i < b2 else 0 for i in x]
    return out


example4_config = SimConfig(
    gt_func=example4_gt,
    gt_risk=example4_risk,
    student_func=func_1b,
    student_optimizer=optimizer_1b_subopt,
    teacher_func=example4_teacher,
    num_train_examples=120,
    num_repeat=20000,
    tag="Example 4"
)


'''
