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

def map_elites(config, n_bins=10, n_init=100, n_iter=500):
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    
    # Define grid for behavior descriptors
    grid = {}
    archive = {}
    
    def get_cell(desc):
        # Normalise descriptor (ex: mean action in [0.6, 1.6] if output ~sigmoid + 0.6)
        # Then map to bins
        d1, d2 = desc
        b1 = int(np.clip((d1 - 0.6) / 1.0 * n_bins, 0, n_bins - 1))
        b2 = int(np.clip((d2) / 1.0 * n_bins, 0, n_bins - 1))
        return (b1, b2)

    # Initial random agents
    for _ in range(n_init):
        agent = Agent(Network, cfg)
        fitness, trajectory = evaluate_with_traj(agent, env, max_steps=cfg["max_steps"])
        desc = agent.descriptor(trajectory)
        cell = get_cell(desc)
        if cell not in archive or archive[cell].fitness < fitness:
            agent.fitness = fitness
            archive[cell] = agent
            grid[cell] = fitness

    # Evolution loop
    for i in tqdm(range(n_iter)):
        if len(archive) == 0:
            continue
        # Select random elite
        parent = np.random.choice(list(archive.values()))
        child_genes = parent.mutate_ga()
        child = Agent(Network, cfg, genes=child_genes)
        fitness, trajectory = evaluate_with_traj(child, env, max_steps=cfg["max_steps"],render=False)
        desc = child.descriptor(trajectory)
        cell = get_cell(desc)
        if cell not in archive or archive[cell].fitness < fitness:
            child.fitness = fitness
            archive[cell] = child
            grid[cell] = fitness

    # Optional: Visualisation
    fitness_map = np.zeros((n_bins, n_bins))
    for (i, j), f in grid.items():
        fitness_map[i, j] = f
    plt.imshow(fitness_map, origin='lower')
    plt.colorbar(label="Fitness")
    plt.title("MAP-Elites Grid")
    plt.xlabel("Descriptor 1")
    plt.ylabel("Descriptor 2")
    plt.show()

    env.close()
    best_cell = max(archive.items(), key=lambda x: x[1].fitness)
    best_agent = best_cell[1]
    return best_agent



#--------------------------------------------------------------------------------------#

def extract_descriptor(obs_start, obs_end):
    # Descripteur : distance horizontale (position x)
    x0 = obs_start[0]
    x1 = obs_end[0]
    distance = x1 - x0
    return [distance]


def map_elites_pyribs(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    sol_dim = len(Agent(Network, cfg).genes)

    # Archive avec 1 seul descripteur : la distance
    archive = CVTArchive(
        solution_dim=sol_dim,
        cells=100,  # 100 cellules
        ranges=[(0, 5)],  # distance de 0 à 5
        samples=10000,
    )

    emitter = GaussianEmitter(
        archive,
        x0=Agent(Network, cfg).genes,
        sigma=cfg["sigma"],
        batch_size=cfg["lambda"],
    )

    best_fitness = -np.inf
    best_genes = None
    fits = []
    total_evals = []

    for gen in tqdm(range(cfg["generations"]), desc="MAP-Elites"):
        solutions = emitter.ask()
        objectives = []
        descriptors = []

        for genes in solutions:
            agent = Agent(Network, cfg, genes=genes)
            env = make_env(cfg["env_name"], robot=cfg["robot"])
            obs_start, _ = env.reset()
            obs = obs_start
            reward = 0
            steps = 0
            done = False

            while not done and steps < cfg["max_steps"]:
                action = agent.act(obs)
                obs, r, done, trunc, _ = env.step(action)
                reward += r
                steps += 1
            env.close()

            desc = extract_descriptor(obs_start, obs)
            objectives.append(reward)
            descriptors.append(desc)

            if reward > best_fitness:
                best_fitness = reward
                best_genes = genes

        archive.add(solutions, objectives, descriptors)
        add_info = [{} for _ in solutions]
        emitter.tell(solutions, objectives, descriptors, add_info)

        fits.append(best_fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))

    # Affiche la progression du meilleur fitness
    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("MAP-Elites Progress")
    plt.show()

    # Renvoyer le meilleur agent
    best_agent = Agent(Network, cfg, genes=best_genes)
    best_agent.fitness = best_fitness
    return best_agent




@ray.remote
def evaluate_mapelite_remote(genes, cfg):
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    agent = Agent(Network, cfg, genes=genes)
    fitness, trajectory = evaluate_with_traj(agent, env, max_steps=cfg["max_steps"])
    desc = agent.descriptor(trajectory)
    env.close()
    return (fitness, desc, genes)

def map_elites_parallel(config, n_bins=10, n_init=100, n_iter=500):
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    def get_cell(desc):
        d1, d2 = desc
        b1 = int(np.clip((d1 - 0.6) / 1.0 * n_bins, 0, n_bins - 1))
        b2 = int(np.clip((d2) / 1.0 * n_bins, 0, n_bins - 1))
        return (b1, b2)

    archive = {}
    grid = {}

    # Phase d'initialisation en parallèle
    initial_agents = [Agent(Network, cfg) for _ in range(n_init)]
    tasks = [evaluate_mapelite_remote.remote(agent.genes, cfg) for agent in initial_agents]
    results = ray.get(tasks)

    for fitness, desc, genes in results:
        cell = get_cell(desc)
        if cell not in archive or archive[cell].fitness < fitness:
            agent = Agent(Network, cfg, genes=genes)
            agent.fitness = fitness
            archive[cell] = agent
            grid[cell] = fitness

    # Boucle principale d'évolution
    for i in tqdm(range(n_iter), desc="MAP-Elites (parallel)"):
        if len(archive) == 0:
            continue
        parents = [np.random.choice(list(archive.values())) for _ in range(cfg["lambda"])]
        child_genes = [parent.mutate_ga() for parent in parents]

        tasks = [evaluate_mapelite_remote.remote(genes, cfg) for genes in child_genes]
        results = ray.get(tasks)

        for fitness, desc, genes in results:
            cell = get_cell(desc)
            if cell not in archive or archive[cell].fitness < fitness:
                agent = Agent(Network, cfg, genes=genes)
                agent.fitness = fitness
                archive[cell] = agent
                grid[cell] = fitness

    # Visualisation
    fitness_map = np.zeros((n_bins, n_bins))
    for (i, j), f in grid.items():
        fitness_map[i, j] = f
    plt.imshow(fitness_map, origin='lower')
    plt.colorbar(label="Fitness")
    plt.title("MAP-Elites Grid (Parallel)")
    plt.xlabel("Descriptor 1")
    plt.ylabel("Descriptor 2")
    plt.show()

    best_cell = max(archive.items(), key=lambda x: x[1].fitness)
    best_agent = best_cell[1]
    return best_agent


@ray.remote
def evaluate_solution_remote(genes, cfg):
    agent = Agent(Network, cfg, genes=genes)
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    obs_start, _ = env.reset()
    obs = obs_start
    reward = 0
    steps = 0
    done = False

    while not done and steps < cfg["max_steps"]:
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        reward += r
        steps += 1

    env.close()
    desc = [obs[0] - obs_start[0]]  # Distance horizontale
    return reward, desc

def map_elites_pyribs_parallelisation(config):
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}

    sol_dim = len(Agent(Network, cfg).genes)

    archive = CVTArchive(
        solution_dim=sol_dim,
        cells=100,
        ranges=[(0, 5)],
        samples=10000,
    )

    emitter = GaussianEmitter(
        archive,
        x0=Agent(Network, cfg).genes,
        sigma=cfg["sigma"],
        batch_size=cfg["lambda"],
    )

    best_fitness = -np.inf
    best_genes = None
    fits = []
    total_evals = []

    for gen in tqdm(range(cfg["generations"]), desc="MAP-Elites"):
        solutions = emitter.ask()

        # Parallélisation des évaluations
        futures = [evaluate_solution_remote.remote(genes, cfg) for genes in solutions]
        results = ray.get(futures)

        objectives = []
        descriptors = []

        for idx, (reward, desc) in enumerate(results):
            objectives.append(reward)
            descriptors.append(desc)

            if reward > best_fitness:
                best_fitness = reward
                best_genes = solutions[idx]

        archive.add(solutions, objectives, descriptors)
        add_info = [{} for _ in solutions]
        emitter.tell(solutions, objectives, descriptors, add_info)

        fits.append(best_fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))

    plt.plot(total_evals, fits)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("MAP-Elites Progress")
    plt.show()

    best_agent = Agent(Network, cfg, genes=best_genes)
    best_agent.fitness = best_fitness
    return best_agent
