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

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter

def walker_descriptors(agent, env, max_steps=500):
    """Calcule les descripteurs pour un agent walker: distance parcourue et hauteur moyenne"""
    obs, _ = env.reset()
    agent.model.reset()
    total_reward = 0
    done = False
    steps = 0
    heights = []
    x_positions = []
    
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
        
        # Extrait la position x et la hauteur depuis info
        # Adapté pour le walker dans Evolution Gym
        if "robot_com" in info:
            x_positions.append(info["robot_com"][0])
            heights.append(info["robot_com"][1])
        steps += 1
    
    x_distance = max(x_positions) - min(x_positions) if x_positions else 0
    mean_height = np.mean(heights) if heights else 0
    
    return np.array([x_distance, mean_height]), total_reward

def map_elites_sans_para(config):
    """Implémentation de MAP-Elites sans utiliser l'optimiseur de pyribs"""
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    
    # Dimensions des descripteurs
    x_bounds = (0, 10)  # Distance parcourue
    h_bounds = (0, 2)   # Hauteur moyenne
    
    # Initialisation de l'agent et récupération de la taille du génome
    initial_agent = Agent(Network, cfg)
    solution_dim = len(initial_agent.genes)
    
    # Création de l'archive
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[20, 20],
        ranges=[x_bounds, h_bounds]
    )
    
    # Emetteur pour générer de nouvelles solutions
    emitter = EvolutionStrategyEmitter(
        archive,
        x0=initial_agent.genes,
        sigma0=cfg["sigma"],
        batch_size=cfg["lambda"]
    )
    
    # Meilleur agent trouvé
    best_agent = Agent(Network, cfg)
    best_agent.fitness = float("-inf")
    
    fits = []
    total_evals = []
    
    bar = tqdm(range(cfg["generations"]), desc="MAP-Elites")
    for gen in bar:
        # Générer de nouvelles solutions
        solutions = emitter.ask()
        
        # Évaluer chaque solution
        objectives = []
        descriptors = []
        for genes in solutions:
            a = Agent(Network, cfg, genes=genes)
            desc, fitness = walker_descriptors(a, env, max_steps=cfg["max_steps"])
            objectives.append(fitness)
            descriptors.append(desc)
            
            # Garder trace du meilleur agent
            if fitness > best_agent.fitness:
                best_agent.genes = genes.copy()
                best_agent.fitness = fitness
        
        # Mettre à jour l'archive
        for i, (solution, objective, desc) in enumerate(zip(solutions, objectives, descriptors)):
            archive.add(np.array([solution]), np.array([objective]), np.array([desc]))

        
        # Informer l'emitter des résultats (sans utiliser optimizer.tell)
        emitter.tell(objectives)
        
        # Suivi des performances
        fits.append(best_agent.fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))
        
        bar.set_description(f"Best: {best_agent.fitness:.2f}, Archive: {len(archive)}")
    
    env.close()
    
    # Visualisation de l'archive
    data = archive.as_pandas()
    plt.figure(figsize=(10, 8))
    plt.scatter(data['index_0'], data['index_1'], c=data['objective'], cmap='viridis', s=40)
    plt.colorbar(label='Fitness')
    plt.xlabel("Distance parcourue")
    plt.ylabel("Hauteur moyenne")
    plt.title("MAP-Elites Archive")
    plt.tight_layout()
    plt.show()
    
    return best_agent
    
#--------------------------------------------------------------------------------------#s
@ray.remote
def evaluate_mapelite_remote(genes, cfg):
    """Fonction d'évaluation distante pour MAP-Elites"""
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    agent = Agent(Network, cfg, genes=genes)
    desc, fitness = walker_descriptors(agent, env, max_steps=cfg["max_steps"])
    env.close()
    return desc, fitness, genes

def map_elites_para(config):
    """Implémentation de MAP-Elites avec parallélisation Ray"""

    
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    
    # Dimensions des descripteurs
    x_bounds = (0, 10)  # Distance parcourue
    h_bounds = (0, 2)   # Hauteur moyenne
    
    # Initialisation de l'agent et récupération de la taille du génome
    initial_agent = Agent(Network, cfg)
    solution_dim = len(initial_agent.genes)
    
    # Création de l'archive
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[20, 20],
        ranges=[x_bounds, h_bounds]
    )
    
    # Emetteur pour générer de nouvelles solutions
    emitter = EvolutionStrategyEmitter(
        archive,
        x0=initial_agent.genes,
        sigma0=cfg["sigma"],
        batch_size=cfg["lambda"]
    )
    
    # Meilleur agent trouvé
    best_agent = Agent(Network, cfg)
    best_agent.fitness = float("-inf")
    
    fits = []
    total_evals = []
    
    bar = tqdm(range(cfg["generations"]), desc="MAP-Elites (Ray)")
    for gen in bar:
        # Générer de nouvelles solutions
        solutions = emitter.ask()
        
        # Évaluation parallèle avec Ray
        tasks = [evaluate_mapelite_remote.remote(genes, cfg) for genes in solutions]
        results = ray.get(tasks)
        
        # Extraire les résultats
        descriptors = [desc for desc, _, _ in results]
        objectives = [fitness for _, fitness, _ in results]
        genes_list = [genes for _, _, genes in results]
        
        # Mettre à jour l'archive et le meilleur agent
        statuses = []
        for i, (solution, objective, desc) in enumerate(zip(solutions, objectives, descriptors)):
            added = archive.add(np.array([solution]), np.array([objective]), np.array([desc]))
            statuses.append(added)
            if objective > best_agent.fitness:
                best_agent.genes = genes_list[i].copy()
                best_agent.fitness = objective

        emitter.tell(
    np.array(solutions),
    np.array(objectives),
    np.array(descriptors),
    {"status": np.array(statuses)}
        )


        
        # Suivi des performances
        fits.append(best_agent.fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))
        
        bar.set_description(f"Best: {best_agent.fitness:.2f}, Archive: {len(archive)}")
    
    # Visualisation de l'archive
    data = archive.as_pandas()
    plt.figure(figsize=(10, 8))
    plt.scatter(data['index_0'], data['index_1'], c=data['objective'], cmap='viridis', s=40)
    plt.colorbar(label='Fitness')
    plt.xlabel("Distance parcourue")
    plt.ylabel("Hauteur moyenne")
    plt.title("MAP-Elites Archive (Parallèle)")
    plt.tight_layout()
    plt.show()
    

    return best_agent


def walker_descriptors_advanced(agent, env, max_steps=500):
    """Descripteurs améliorés: distance parcourue et stabilité (écart-type de hauteur)"""
    obs, _ = env.reset()
    agent.model.reset()
    total_reward = 0
    done = False
    steps = 0
    heights = []
    x_positions = []
    
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
        
        if "robot_com" in info:
            x_positions.append(info["robot_com"][0])
            heights.append(info["robot_com"][1])
        steps += 1
    
    # Premier descripteur: distance parcourue
    x_distance = max(x_positions) - min(x_positions) if x_positions else 0
    
    # Deuxième descripteur: stabilité (inverse de l'écart-type de la hauteur)
    # Un écart-type faible signifie une marche stable
    height_std = np.std(heights) if len(heights) > 1 else 0
    stability = 1.0 / (1.0 + height_std)  # Normalisation: plus c'est stable, plus c'est proche de 1
    
    return np.array([x_distance, stability]), total_reward

def walker_descriptors_expert(agent, env, max_steps=500):
    """Descripteurs experts: distance et fréquence des pas (oscillations)"""
    obs, _ = env.reset()
    agent.model.reset()
    total_reward = 0
    done = False
    steps = 0
    heights = []
    x_positions = []
    
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
        
        if "robot_com" in info:
            x_positions.append(info["robot_com"][0])
            heights.append(info["robot_com"][1])
        steps += 1
    
    # Distance parcourue
    x_distance = max(x_positions) - min(x_positions) if x_positions else 0
    
    # Fréquence des pas (compter les oscillations verticales)
    step_frequency = 0
    if len(heights) > 3:
        # Détection de pics (maxima locaux)
        peaks = 0
        for i in range(1, len(heights)-1):
            if heights[i] > heights[i-1] and heights[i] > heights[i+1]:
                peaks += 1
        step_frequency = peaks / len(heights)  # Normaliser par rapport au temps
    
    return np.array([x_distance, step_frequency]), total_reward


@ray.remote
def evaluate_mapelite_advanced_remote(genes, cfg):
    """Fonction d'évaluation distante pour MAP-Elites avec descripteurs avancés"""
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    agent = Agent(Network, cfg, genes=genes)
    desc, fitness = walker_descriptors_advanced(agent, env, max_steps=cfg["max_steps"])
    env.close()
    return desc, fitness, genes

def map_elites_para_advanced(config):
    """Implémentation de MAP-Elites avec parallélisation Ray et descripteurs avancés"""

    
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    
    # Dimensions des descripteurs
    x_bounds = (0, 10)  # Distance parcourue
    stability_bounds = (0, 1)  # Stabilité (normalisation inversée de l'écart-type)
    
    # Initialisation de l'agent et récupération de la taille du génome
    initial_agent = Agent(Network, cfg)
    solution_dim = len(initial_agent.genes)
    
    # Création de l'archive
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[20, 20],
        ranges=[x_bounds, stability_bounds]
    )
    
    # Emetteur pour générer de nouvelles solutions
    emitter = EvolutionStrategyEmitter(
        archive,
        x0=initial_agent.genes,
        sigma0=cfg["sigma"],
        batch_size=cfg["lambda"]
    )
    
    # Meilleur agent trouvé
    best_agent = Agent(Network, cfg)
    best_agent.fitness = float("-inf")
    
    fits = []
    total_evals = []
    
    bar = tqdm(range(cfg["generations"]), desc="MAP-Elites Avancé (Ray)")
    for gen in bar:
        # Générer de nouvelles solutions
        solutions = emitter.ask()
        
        # Évaluation parallèle avec Ray
        tasks = [evaluate_mapelite_advanced_remote.remote(genes, cfg) for genes in solutions]
        results = ray.get(tasks)
        
        # Extraire les résultats
        descriptors = [desc for desc, _, _ in results]
        objectives = [fitness for _, fitness, _ in results]
        genes_list = [genes for _, _, genes in results]
        
        # Mettre à jour l'archive et le meilleur agent
        for i, (solution, objective, desc) in enumerate(zip(solutions, objectives, descriptors)):
            archive.add(np.array([solution]), np.array([objective]), np.array([desc]))
            if objective > best_agent.fitness:
                best_agent.genes = genes_list[i].copy()
                best_agent.fitness = objective
        
        # Informer l'emitter des résultats
        emitter.tell(objectives)
        
        # Suivi des performances
        fits.append(best_agent.fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))
        
        bar.set_description(f"Best: {best_agent.fitness:.2f}, Archive: {len(archive)}")
    
    # Visualisation de l'archive
    data = archive.as_pandas()
    plt.figure(figsize=(10, 8))
    plt.scatter(data['index_0'], data['index_1'], c=data['objective'], cmap='viridis', s=40)
    plt.colorbar(label='Fitness')
    plt.xlabel("Distance parcourue")
    plt.ylabel("Stabilité")
    plt.title("MAP-Elites Archive (Descripteurs Avancés)")
    plt.tight_layout()
    plt.show()

    return best_agent



@ray.remote
def evaluate_mapelite_expert_remote(genes, cfg):
    """Fonction d'évaluation distante pour MAP-Elites avec descripteurs experts"""
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    agent = Agent(Network, cfg, genes=genes)
    desc, fitness = walker_descriptors_expert(agent, env, max_steps=cfg["max_steps"])
    env.close()
    return desc, fitness, genes

def map_elites_para_expert(config):
    """Implémentation de MAP-Elites avec parallélisation Ray et descripteurs experts"""
    
    cfg = get_cfg(config["env_name"], robot=config["robot"])
    cfg = {**config, **cfg}
    
    # Dimensions des descripteurs
    x_bounds = (0, 10)  # Distance parcourue
    frequency_bounds = (0, 0.5)  # Fréquence des pas (oscillations)
    
    # Initialisation de l'agent et récupération de la taille du génome
    initial_agent = Agent(Network, cfg)
    solution_dim = len(initial_agent.genes)
    
    # Création de l'archive
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[20, 20],
        ranges=[x_bounds, frequency_bounds]
    )
    
    # Emetteur pour générer de nouvelles solutions
    emitter = EvolutionStrategyEmitter(
        archive,
        x0=initial_agent.genes,
        sigma0=cfg["sigma"],
        batch_size=cfg["lambda"]
    )
    
    # Meilleur agent trouvé
    best_agent = Agent(Network, cfg)
    best_agent.fitness = float("-inf")
    
    fits = []
    total_evals = []
    
    bar = tqdm(range(cfg["generations"]), desc="MAP-Elites Expert (Ray)")
    for gen in bar:
        # Générer de nouvelles solutions
        solutions = emitter.ask()
        
        # Évaluation parallèle avec Ray
        tasks = [evaluate_mapelite_expert_remote.remote(genes, cfg) for genes in solutions]
        results = ray.get(tasks)
        
        # Extraire les résultats
        descriptors = [desc for desc, _, _ in results]
        objectives = [fitness for _, fitness, _ in results]
        genes_list = [genes for _, _, genes in results]
        
        # Mettre à jour l'archive et le meilleur agent
        for i, (solution, objective, desc) in enumerate(zip(solutions, objectives, descriptors)):
            archive.add(np.array([solution]), np.array([objective]), np.array([desc]))
            if objective > best_agent.fitness:
                best_agent.genes = genes_list[i].copy()
                best_agent.fitness = objective
        
        # Informer l'emitter des résultats
        emitter.tell(objectives)
        
        # Suivi des performances
        fits.append(best_agent.fitness)
        total_evals.append(cfg["lambda"] * (gen + 1))
        
        bar.set_description(f"Best: {best_agent.fitness:.2f}, Archive: {len(archive)}")
    
    # Visualisation de l'archive
    data = archive.as_pandas()
    plt.figure(figsize=(10, 8))
    plt.scatter(data['index_0'], data['index_1'], c=data['objective'], cmap='viridis', s=40)
    plt.colorbar(label='Fitness')
    plt.xlabel("Distance parcourue")
    plt.ylabel("Fréquence des pas")
    plt.title("MAP-Elites Archive (Descripteurs Experts)")
    plt.tight_layout()
    plt.show()
    
    return best_agent