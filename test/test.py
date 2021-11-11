import numpy as np
from src.function_models import BinaryFunction

x = np.random.uniform(low=0, high=1, size=(2,))
x = [0.1, 0.3, 0.5, 0.9]
params = [0.2, 0.4, 0.8]
gt = BinaryFunction(params)
loss, opt_params = BinaryFunction.get_empirical_risk(gt, x, 2)