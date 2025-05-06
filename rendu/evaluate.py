from environment import *
from importations import *
from agent import * 
from para import *
from algos import *
import json 

"""thrower = np.array(
    [[4, 0, 4, 1, 4],
     [3, 0, 4, 0, 3],
     [4, 0, 4, 2, 0],
     [4, 3, 4, 4, 2],
     [0, 4, 1, 0, 0]])


config = {
        "env_name": "Thrower-v0",
        "robot": thrower,
        "generations": 5,
        "lambda": 2,  # CMA-ES population size
        "sigma": 0.5,  # Initial mutation std
        "max_steps": 500,
    }

cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
cfg = {**config, **cfg} # Merge configs"""


for name in ["thrower","climber","walker"]:
    file=name+".json"
    a = load_solution(file)
    cfg = a.config
    env = make_env(cfg["env_name"], robot=cfg["robot"], render_mode="rgb_array")

    fitness = evaluate(a, env)
    env.close()
    print("Fitness pour le "+ name + ": " +str(fitness))


