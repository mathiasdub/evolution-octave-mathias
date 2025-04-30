from evogym import sample_robot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
from evogym.utils import get_full_connectivity
from tqdm import tqdm
import json
import time
import imageio
from agent import *
from environment import *



walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])


config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 500, # To change: increase!
    "lambda": 5,
    "max_steps": 500, # to change to 500
}

cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
cfg = {**config, **cfg} # Merge configs

def load_solution(cfg,name="solution.json"):
    with open(name, "r") as f:
        cfg = json.load(f)
    cfg["robot"] = np.array(cfg["robot"])
    cfg["genes"] = np.array(cfg["genes"])
    a = Agent(Network, cfg, genes=cfg["genes"])
    a.fitness = cfg["fitness"]
    return a

a = load_solution(cfg=cfg,name="solution.json")
cfg = a.config
env = make_env(config["env_name"], robot=config["robot"], render_mode="rgb_array")
env.metadata.update({'render_modes': ["rgb_array"]})

a.fitness, imgs = evaluate(a, env, render=True)
env.close()
print(a.fitness)

imageio.mimsave(f'gif/Walker.gif', imgs, duration=(1/50.0))