from importations import *
from environment import *


from para import *




#--------------------------------------------------------------------------------------#


def one_plus_lambda(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Elite
    elite = Agent(Network, cfg)

    elite.fitness = evaluate(elite, env, max_steps=cfg["max_steps"])

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = [Agent(Network, cfg, genes=elite.mutate_ga()) 
            for _ in range(cfg["lambda"])]


        pop_fitness = [evaluate(a,env, max_steps=cfg["max_steps"]) for a in population]

        best = np.argmax(pop_fitness)
        best_fit = pop_fitness[best]
        if best_fit > elite.fitness:
            elite.genes = population[best].genes
            elite.fitness = best_fit
        fits.append(elite.fitness)
        total_evals.append(len(population) * (gen+1))
        bar.set_description(f"Best: {elite.fitness}")
        
    env.close()
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.show()
    return elite


#--------------------------------------------------------------------------------------#



def ES(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    
    # Update weights
    mu = cfg["mu"]
    w = np.array([np.log(mu + 0.5) - np.log(i)
                          for i in range(1, mu + 1)])
    w /= np.sum(w)
    
    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Center of the distribution
    elite = Agent(Network, cfg)
    elite.fitness = -np.inf
    theta = elite.genes
    d = len(theta)

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        population = []
        for i in range(cfg["lambda"]):
            genes = theta + np.random.randn(len(theta)) * cfg["sigma"]
            ind = Agent(Network, cfg, genes=genes)
            # ind.fitness = evaluate(ind, env, max_steps=cfg["max_steps"])
            population.append(ind)

        # with Pool(processes=len(population)) as pool:
        #     pop_fitness = pool.starmap(mp_eval, [(a, cfg) for a in population])
        
        pop_fitness = [evaluate(a, env, max_steps=cfg["max_steps"]) for a in population]
        
        for i in range(len(population)):
            population[i].fitness = pop_fitness[i]

        # sort by fitness
        inv_fitnesses = [- f for f in pop_fitness]
        # indices from highest fitness to lowest
        idx = np.argsort(inv_fitnesses)
        
        step = np.zeros(d)
        for i in range(mu):
            # update step
            step = step + w[i] * (population[idx[i]].genes - theta)
        # update theta
        theta = theta + step * cfg["lr"]

        if pop_fitness[idx[0]] > elite.fitness:
            elite.genes = population[idx[0]].genes
            elite.fitness = pop_fitness[idx[0]]

        fits.append(elite.fitness)
        total_evals.append(len(population) * (gen+1))

        bar.set_description(f"Best: {elite.fitness}")
        
    env.close()
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.show()
    return elite

def save_solution(a, cfg, name="solution.json"):
    save_cfg = {}
    for i in ["env_name", "robot", "n_in", "h_size", "n_out"]:
        assert i in cfg, f"{i} not in config"
        save_cfg[i] = cfg[i]
    save_cfg["robot"] = cfg["robot"].tolist()
    save_cfg["genes"] = a.genes.tolist()
    save_cfg["fitness"] = float(a.fitness)
    # save
    with open(name, "w") as f:
        json.dump(save_cfg, f)
    return save_cfg

def load_solution(name="solution.json"):
    with open(name, "r") as f:
        cfg = json.load(f)
    cfg["robot"] = np.array(cfg["robot"])
    cfg["genes"] = np.array(cfg["genes"])
    a = Agent(Network, cfg, genes=cfg["genes"])
    a.fitness = cfg["fitness"]
    return a


#--------------------------------------------------------------------------------------

#ray.init(ignore_reinit_error=True)  # Réinitialiser Ray

@ray.remote
def evaluate_agent_remote(genes, cfg):
    env = EvoGymEnv(cfg["env_name"], cfg["robot"])
    agent = Agent(Network, cfg, genes=genes)
    fitness = evaluate(agent, env, max_steps=cfg["max_steps"])
    return -fitness  # CMA-ES minimise

def CMA_ES(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"])  # Get network dims
    cfg = {**config, **cfg}  # Merge configs

    agent = Agent(Network, cfg)
    theta = agent.genes  # initial parameters

    es = cma.CMAEvolutionStrategy(theta, cfg["sigma"], {
        'popsize': cfg["lambda"]
    })

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        solutions = es.ask()

        # Évaluation parallèle avec Ray
        tasks = [evaluate_agent_remote.remote(genes, cfg) for genes in solutions]
        fitnesses = ray.get(tasks)  # List of negated fitnesses

        es.tell(solutions, fitnesses)
        best_idx = np.argmin(fitnesses)
        best_fitness = -fitnesses[best_idx]
        best_genes = solutions[best_idx]

        if agent.fitness is None or best_fitness > agent.fitness:
            agent.genes = best_genes
            agent.fitness = best_fitness

        fits.append(agent.fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))
        bar.set_description(f"Best: {agent.fitness}")

    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("CMA-ES Progress")
    plt.show()

    return agent



def CMA_ES_sans_para(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"])  # Get network dims
    cfg = {**config, **cfg}  # Merge configs

    env = make_env(cfg["env_name"], robot=cfg["robot"])
    agent = Agent(Network, cfg)
    theta = agent.genes  # initial parameters

    es = cma.CMAEvolutionStrategy(theta, cfg["sigma"], {
        'popsize': cfg["lambda"]
    })

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        solutions = es.ask()
        fitnesses = []
        for genes in solutions:
            a = Agent(Network, cfg, genes=genes)
            f = evaluate(a, env, max_steps=cfg["max_steps"])
            fitnesses.append(-f)  # CMA-ES minimizes, so we negate

        es.tell(solutions, fitnesses)
        best_idx = np.argmin(fitnesses)
        best_fitness = -fitnesses[best_idx]
        best_genes = solutions[best_idx]

        if agent.fitness is None or best_fitness > agent.fitness:
            agent.genes = best_genes
            agent.fitness = best_fitness

        fits.append(agent.fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))
        bar.set_description(f"Best: {agent.fitness}")

    env.close()
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("CMA-ES Progress")
    plt.show()
    return agent
    
#--------------------------------------------------------------------------------------#

"""@ray.remote
def evaluate_remote(genes, cfg):
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    a = Agent(Network, cfg, genes=genes)
    fitness = evaluate(a, env, max_steps=cfg["max_steps"])
    desc = descriptor(a, env)
    env.close()
    return fitness, desc

def descriptor(agent, env):
    obs, _ = env.reset()
    agent.model.reset()
    done = False
    total_steps = 0
    trajectory = []

    while not done and total_steps < 500:
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        trajectory.append(obs)
        total_steps += 1

    trajectory = np.array(trajectory)
    if trajectory.shape[0] < 1:
        return np.zeros(2)
    desc = np.mean(trajectory[:, :2], axis=0)
    return desc

def MAP_Elites(config):
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    
    archive = GridArchive(
        dims=[10, 10],
        ranges=[[-1, 1], [-1, 1]],
        seed=42
    )

    emitters = [
        GaussianEmitter(
            archive,
            x0=Agent(Network, cfg).genes,
            sigma=cfg["sigma"],
            bounds=None,
        )
        for _ in range(5)
    ]

    optimizer = Optimizer(archive, emitters)

    fits = []
    total_evals = []

    bar = tqdm(range(cfg["generations"]))
    for gen in bar:
        solutions = optimizer.ask()

        # Parallel eval with Ray
        futures = [evaluate_remote.remote(genes, cfg) for genes in solutions]
        results = ray.get(futures)

        fitnesses, descriptors = zip(*results)

        optimizer.tell(solutions, descriptors, fitnesses)

        best_index = np.argmax(fitnesses)
        best_fit = fitnesses[best_index]
        fits.append(best_fit)
        total_evals.append(cfg["lambda"] * (gen + 1))
        bar.set_description(f"Best fitness: {best_fit:.2f}")

    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("MAP-Elites Progress")
    plt.show()

    best_idx = np.argmax(archive._objective_values)
    best_solution = archive._solutions[best_idx]
    best_fitness = archive._objective_values[best_idx]

    best_agent = Agent(Network, cfg, genes=best_solution)
    best_agent.fitness = best_fitness

    ray.shutdown()
    return best_agent"""


#--------------------------------------------------------------------------------------#