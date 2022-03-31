from simulations.st_simulation import Simulation
from simulations.configs import example_theory_check8
import logging
from datetime import datetime

if __name__ == '__main__':
    conf = example_theory_check8
    sim = Simulation(conf, multi_proc=True, log=True)
    print(sim.sim_config)
    sim.run()
    #plt.show()
 