from simulations.st_simulation import Simulation, Simulation2
from simulations.configs import CONFIGS


if __name__ == '__main__':
    conf = CONFIGS['example_theory_check1']
    sim = Simulation2(conf, multi_proc=True, log=False)
    print(sim.sim_config)
    sim.run()

    conf = CONFIGS['example_theory_check2']
    sim = Simulation2(conf, multi_proc=True, log=False)
    print(sim.sim_config)
    sim.run()

