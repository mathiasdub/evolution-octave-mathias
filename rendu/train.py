from  algos import*


#------------ Choix du robot --------------------------------------------------#
robot="climber"
#robot="walker"
#robot="thrower"
#-------------------------------------------------------------------------------#

climber = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 3, 3, 3],
    [4, 2, 4, 1, 4],
    [0, 0, 3, 4, 0],
    [3, 3, 1, 1, 3]
    ])
walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 1, 1]
    ])
thrower = np.array([
    [4, 0, 4, 1, 4], 
    [3, 0, 4, 0, 3], 
    [4, 0, 4, 2, 0], 
    [4, 3, 4, 4, 2], 
    [0, 4, 1, 0, 0]
    ])

if robot=="thrower":
    config = {
        "env_name": "Thrower-v0",
        "robot": thrower,
        "generations": 200,
        "lambda":50,  # CMA-ES population size
        "sigma": 4,  # Initial mutation std
        "max_steps": 300,
    }
elif robot=="climber":
    config = {
        "env_name": "Climber-v2",
        "robot": climber,
        "generations": 200,
        "lambda":50,  # CMA-ES population size
        "sigma": 4,  # Initial mutation std
        "max_steps": 500,
    }   
else:
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

#---------- CMAES pour walker et thrower / ES pour le climber ------------------#
a = CMA_ES(config)
#a = ES(config)
#-------------------------------------------------------------------------------#

a.fitness
    


