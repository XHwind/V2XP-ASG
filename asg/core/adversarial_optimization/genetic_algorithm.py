import copy

from asg.core.adversarial_optimization.base_adversarial import *
from asg.utils.adversarial_utils import softmax

class GeneticAlgorithm(BaseAdversarial):
    """
    A class for Genetic Algorithm.

    Args:
        model: nn.module
            Perception module.
        config: dict
            The configuration dictionary for adversarial algorithm.

    Attributes:
        epsilon: float
            Update step for the search direction.
        max_iter: int
            The maximum number of iteration.
    """

    def __init__(self, opt, config):
        super(GeneticAlgorithm, self).__init__(opt, config)
        self.max_iter = config["max_iter"]

        # BO specific parameters
        self.pop_size = config["GA"]["population_size"]
        self.num_generations = config["GA"]["num_generations"]
        self.num_max_plateaus = config["GA"]["num_max_plateaus"]
        self.tau = config["GA"]["tau"]

        # mutation related adaptive parameters
        self.mutation_p_min = config["GA"]["mutation"]["p_min"]
        self.mutation_p_initial = config["GA"]["mutation"]["p_initial"]
        self.mutation_z_min = config["GA"]["mutation"]["min_range"]
        self.mutation_z_initial = config["GA"]["mutation"]["initial_range"]
        self.gamma = config["GA"]["gamma"]

    def init_model(self, scene, Q, vid_list, cav_id_list, rsu_id_list):
        """
        Initialize the bayesian optimization models.
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all valid perturbations.
            vid_list: list
                List of vehicle IDs to perturb
            cav_id_list: list
                List of selected CAV IDs.
            rsu_id_list: list
                List of RSU IDs.

        Returns:
            score_best: float
                Min score found so far
            q_best: dict
                Best perturbation associated with score_best so far with format {vid: carla.Transform  for vid in vid_list}.

        """

        # Get bounds for the perturbation

        pop = []
        for i in range(self.pop_size):
            q = random.choice(Q)
            q_np = self.convert_q_to_numpy(q)
            pop.append({"q": q, "q_np": q_np})
        self.evaluate_populations(scene, pop, cav_id_list, rsu_id_list)
        return pop

    def softmax(self, x):
        if isinstance(x, list):
            x = np.array(x)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def evaluate_populations(self, scene, pop, cav_id_list, rsu_id_list):
        for parent in pop:
            score = self.perturb_agents(scene, parent["q"], cav_id_list,
                                        rsu_id_list)
            parent["score"] = score

    def compute_fitness(self, scene, pop, cav_id_list, rsu_id_list):

        for parent in pop:
            parent["fitness"] = 1.0 / max(parent["score"].adv_loss, 0.1)

        select_probs = softmax([p["fitness"] / self.tau for p in pop])

        for i, parent in enumerate(pop):
            parent["select_prob"] = select_probs[i]
        return pop

    def crossover_and_mutation(self, parent1, parent2, p, z):

        vid_list = sorted(parent1["q"].keys())

        # Convert carla Transform to numpy array for manipulations
        q1_np = self.convert_q_to_numpy(parent1["q"])
        q2_np = self.convert_q_to_numpy(parent2["q"])

        # Crossover
        # Select parent1 and parent2's features with probability (p_select, 1-p_select)
        p_select = parent1["fitness"] / (
                parent1["fitness"] + parent2["fitness"])
        xover_prob = np.random.uniform(0, 1, q1_np.shape) < p_select
        xover_prob = xover_prob.astype(float)
        q_child_np = q1_np * xover_prob + q2_np * (1 - xover_prob)

        # mutation
        mutation_noise = np.random.uniform(z[0], z[1], q_child_np.shape)
        mutation_mask = np.random.uniform(0, 1, q_child_np.shape) < p
        mutation_mask = mutation_mask.astype(float)
        q_child_np_mutated = q_child_np + mutation_mask * mutation_noise

        # clipping
        pbounds = self.get_bounds(vid_list)
        q_child_np_mutated = np.clip(q_child_np_mutated, pbounds["min"],
                                     pbounds["max"])

        child = {}
        q_child = self.convert_q_np_to_q(q_child_np_mutated, vid_list)
        child["q"] = q_child
        child["q_np"] = q_child_np_mutated
        return child

    def update_ga_parameters(self, parent_pop, child_pop, num_plateaus, p, z):
        parent_best_score = min(parent["score"] for parent in parent_pop)
        child_best_score = min(child["score"] for child in child_pop)
        if parent_best_score <= child_best_score:
            num_plateaus += 1
            num_plateaus = min(num_plateaus, self.num_max_plateaus)
        p = max(self.mutation_p_min,
                self.mutation_p_initial * (self.gamma ** num_plateaus))
        z[0] = min(self.mutation_z_min[0],
                   self.mutation_z_initial[0] * (self.gamma ** num_plateaus))
        z[1] = max(self.mutation_z_min[1],
                   self.mutation_z_initial[1] * (self.gamma ** num_plateaus))
        return num_plateaus, p, z

    def project_chromosome_to_Q(self, chrom, Q_np, vid_list):
        q = chrom["q"]
        q_proj = self.project_q_to_Q(q, Q_np, vid_list)
        chrom_proj = {
            "q": q_proj,
            "q_np": self.convert_q_to_numpy(q_proj)
        }
        return chrom_proj

    @register_agent_score_drop
    def forward(self, scene, i, vis_flag=True):
        """
        Search adversarial scenes.

        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            i: int
                Current scene index.
            vis_flag: boolean
                True if visualize the lidar, gt and predictions.

        Returns:
            out: dict
                Adversarial attack model's output.
        """
        rsu_id_list = self.select_rsu(i, scene)
        cav_id_list, score_cav_selection, score_first, weight_dict = \
            self.select_adversarial_cavs_attn(rsu_id_list,
                                              scene,
                                              vis_flag=vis_flag)
        if vis_flag:
            score = self.get_adversarial_cost(scene, cav_id_list,
                                              rsu_id_list, vis_flag=vis_flag,
                                              vis_save_path="./cav_selection.png")

        vid_list, Q = self.get_vid_list_and_feasible_set(scene,
                                                         rsu_id_list,
                                                         cav_id_list)
        Q_np = self.convert_Q_to_numpy(Q)
        vid_list = sorted(vid_list)

        pop = self.init_model(scene, Q, vid_list, cav_id_list, rsu_id_list)

        p, z = self.mutation_p_initial, copy.copy(self.mutation_z_initial)
        num_plateaus = 0

        score_min = min(p["score"] for p in pop)
        q_best = min(pop, key=lambda x: x["score"])[
            "q"] if score_min < score_cav_selection else None
        # Ensure it is the smallest among all the observed scores
        score_min = min(score_min, score_cav_selection)

        for g in range(self.num_generations):
            self.compute_fitness(scene, pop, cav_id_list, rsu_id_list)
            parent_elite = min(pop, key=lambda x: x["score"])
            score_elite = min(p["score"] for p in pop)
            new_pop = [parent_elite]
            for i in range(self.pop_size - 1):
                weights = [p["select_prob"] for p in pop]
                parent1, parent2 = random.choices(pop, k=2, weights=weights)
                child_old = self.crossover_and_mutation(parent1, parent2, p, z)
                child = self.project_chromosome_to_Q(child_old, Q_np, vid_list)
                new_pop.append(child)
            self.evaluate_populations(scene, new_pop, cav_id_list, rsu_id_list)
            num_plateaus, p, z = self.update_ga_parameters(pop, new_pop,
                                                           num_plateaus, p, z)
            child_score = min(child["score"] for child in new_pop)
            print("parent_score", score_elite, "child_score", child_score)
            pop = new_pop
            if score_elite < score_min:
                score_min = score_elite
                q_best = min(pop, key=lambda x: x["score"])["q"]

        # self.register_agent_score_drop(q_best, cav_selection_score - score_min)
        if vis_flag:
            self.get_adversarial_cost(scene, cav_id_list, rsu_id_list,
                                      vis_flag=vis_flag,
                                      vis_save_path="./adversarial_position.png")
        out = {
            "score_first": score_first,
            "score_cav_selection": score_cav_selection,
            "score_best": score_min,
            "cav_id_list": cav_id_list,
            "rsu_id_list": rsu_id_list,
            "q_best": q_best,
            "weight_dict": weight_dict
        }
        return out
