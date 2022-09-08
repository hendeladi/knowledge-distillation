from simulations.st_simulation import Simulation, Simulation2
from simulations.configs import CONFIGS


if __name__ == '__main__':
    conf = CONFIGS['example1']
    sim = Simulation(conf, multi_proc=True, log=True)
    print(sim.sim_config)
    sim.run()


