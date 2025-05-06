from  algos import*
import os
from gymnasium.wrappers import RecordVideo
import imageio.v2 as imageio  # Compatible avec imageio>=2.9
import sys



#algo='one_plus_lambda'
algo="ES"
#algo="CMA_ES"
#algo="MAP_Elites"

walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 0, 0, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])


if algo=='one_plus_lambda':
    config = {
        "env_name": "Walker-v0",
        "robot": walker,
        "generations": 100, # To change: increase!
        "lambda": 10,
        "max_steps": 100, # to change to 500
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs

    a = one_plus_lambda(config)
    save_solution(a, cfg, "onepluslmabda_100_10.json")
    a.fitness



    env = make_env(env_name=config["env_name"], robot=config["robot"])
    
    
    
if algo=="ES":
    config = {    
        "env_name": "Walker-v0",
        "robot": walker,
        "generations": 100, # to change: increase!
        "lambda": 10, # Population size
        "mu": 5, # Parents pop size
        "sigma": 0.1, # mutation std
        "lr": 1, # Learning rate
        "max_steps": 200, # to change to 500
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs

    a = ES(config)
    save_solution(a, cfg, "es_100_10.json")
    a.fitness



    env = make_env(config["env_name"], robot=config["robot"])
    evaluate(a, env, render=False)
    env.close()


if algo == "CMA_ES":
    config = {
        "env_name": "Walker-v0",
        "robot": walker,
        "generations": 500,
        "lambda":20,  # CMA-ES population size
        "sigma": 2,  # Initial mutation std
        "max_steps": 500,
    }
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    a = CMA_ES(config)
    save_solution(a, cfg,"walkerv2_avec_para_500_20_2.json")
    a.fitness
    
    env = make_env(config["env_name"], robot=config["robot"])
    evaluate(a, env, render=False)
    env.close()


if algo == "MAP_Elites":
    config = {
        "env_name": "Walker-v0",
        "robot": walker,
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