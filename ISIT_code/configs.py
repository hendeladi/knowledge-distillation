
from sets import Region
from function_models import BinaryFunction


AllPlots = ["risk", "risk_std", "emp_risk", "gen_error", "parameters", "delta_far_prob", "delta_far_prob_term1",
            "delta_far_prob_term2"]
DefaultPlots = ["risk", "risk_std", "parameters", "delta_far_prob", "main_term"]


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


example1_config = SimConfig(
    gt_func=BinaryFunction([0.6, 0.9]),
    student_num_params=1,
    num_train_examples=list(range(4, 120, 2)),
    num_repeat=50000,
    delta=0.1,
    Aopt={"gt": Region([(0, 0.9)]), "kd": Region([(0, 0.9)])},
    tag="",
    dest_dir=""
)



CONFIGS = {
    'example1': example1_config
}