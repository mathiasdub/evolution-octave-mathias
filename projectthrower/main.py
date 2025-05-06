from  algos import*
import os
from gymnasium.wrappers import RecordVideo
import imageio.v2 as imageio  # Compatible avec imageio>=2.9
import sys



#algo='one_plus_lambda'
algo="ES"
#algo="CMA_ES"
#algo="MAP_Elites"


thrower = np.array([
    [0, 0, 3, 1, 3],
    [2, 0, 4, 0, 1],
    [1, 0, 4, 0, 0],
    [1, 3, 4, 2, 0],
    [0, 1, 4, 0, 0]
    ])


if algo=='one_plus_lambda':
    config = {
        "env_name": "thrower-v0",
        "robot": thrower,
        "generations": 100, # To change: increase!
        "lambda": 10,
        "max_steps": 100, # to change to 500
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs

    a = one_plus_lambda(config)
    save_solution(a, cfg)
    a.fitness



    env = make_env(env_name=config["env_name"], robot=config["robot"])
    
    
    
if algo=="ES":
    config = {    "env_name": "Thrower-v0",
        "robot": thrower,
        "generations": 500, # to change: increase!
        "lambda": 20, # Population size
        "mu": 5, # Parents pop size
        "sigma": 1, # mutation std
        "lr": 1, # Learning rate
        "max_steps": 300, # to change to 500
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs

    a = ES(config)
    save_solution(a, cfg, "ES_thrower_500_20_5_1.json")
    a.fitness



    env = make_env(config["env_name"], robot=config["robot"])
    evaluate(a, env, render=False)
    env.close()


if algo == "CMA_ES":
    config = {
        "env_name": "Thrower-v0",
        "robot": thrower,
        "generations": 200,
        "lambda":50,  # CMA-ES population size
        "sigma": 4,  # Initial mutation std
        "max_steps": 500,
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    a = CMA_ES(config)
    save_solution(a, cfg,"cmaes_avec_para_200_50_4.json")
    a.fitness
    
    env = make_env(config["env_name"], robot=config["robot"])
    evaluate(a, env, render=False)
    env.close()


if algo == "MAP_Elites":
    config = {
        "env_name": "thrower-v0",
        "robot": thrower,
        "generations": 3,
        "lambda": 3,  # nombre de solutions par génération
        "sigma": 0.5,
        "max_steps": 500,
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    a = MAP_Elites(config)
    agent_data = {
        "genes": a.genes.tolist(),  # convertit le tableau numpy en liste
        "fitness": a.fitness
    }

    with open("best_agent.json", "w") as f:
        json.dump(agent_data, f, indent=4)