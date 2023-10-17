from bayes_opt import BayesianOptimization as BO
from bayes_opt import UtilityFunction
from asg.core.adversarial_optimization.base_adversarial import *


class BayesianOptimization(BaseAdversarial):
    """
    A class for Bayesian Optimization algorithm.

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
        super(BayesianOptimization, self).__init__(opt, config)
        self.max_iter = config["max_iter"]

        # BO specific parameters
        self.num_initial_samples = config["BO"]["num_initial_samples"]
        self.opt_random_state = config["BO"]["optimizer"]["random_state"]
        self.utility_params = config["BO"]["utility"]
        self.normalize_flag = config["BO"]["normalize_flag"]

    def get_pbounds(self, vid_list):
        if not self.normalize_flag:
            x_factor, y_factor, yaw_factor = 1, 1, 1
        else:
            x_factor, y_factor, yaw_factor = abs(self.x_range[1]), \
                abs(self.y_range[1]), \
                abs(self.perturb_angle_range[1])
        pbounds = {}
        for vid in vid_list:
            var_name = "agent_" + str(vid)
            pbounds.update({
                var_name + "_x": (self.x_range[0] / x_factor, \
                                  self.x_range[1] / x_factor),
                var_name + "_y": (self.y_range[0] / y_factor, \
                                  self.y_range[1] / y_factor),
                var_name + "_yaw": (self.perturb_angle_range[0] / yaw_factor,
                                    self.perturb_angle_range[1] / yaw_factor)
            })
        return pbounds

    def convert_q_to_BO_format(self, q):
        if not self.normalize_flag:
            x_factor, y_factor, yaw_factor = 1, 1, 1
        else:
            x_factor, y_factor, yaw_factor = abs(self.x_range[1]), \
                abs(self.y_range[1]), \
                abs(self.perturb_angle_range[1])
        q_bo = {}
        for i, (vid, q_agent) in enumerate(q.items()):
            var_name = "agent_" + str(vid)
            q_bo.update({
                var_name + "_x": q[vid].location.x / x_factor,
                var_name + "_y": q[vid].location.y / y_factor,
                var_name + "_yaw": q[vid].rotation.yaw / yaw_factor
            })
        return q_bo

    def convert_q_bo_to_carla_format(self, q_bo):
        if not self.normalize_flag:
            x_factor, y_factor, yaw_factor = 1, 1, 1
        else:
            x_factor, y_factor, yaw_factor = abs(self.x_range[1]), \
                abs(self.y_range[1]), \
                abs(self.perturb_angle_range[1])
        q = {}
        vid_list = set([int(k.split("_")[1]) for k in q_bo.keys()])

        for vid in vid_list:
            var_name = "agent_" + str(vid)
            q_loc = carla.Location(q_bo[var_name + "_x"] * x_factor,
                                   q_bo[var_name + "_y"] * y_factor)
            q_rot = carla.Rotation(yaw=(q_bo[var_name + "_yaw"] * yaw_factor))

            q.update({
                vid: carla.Transform(location=q_loc, rotation=q_rot)
            })

        return q

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
        pbounds = self.get_pbounds(vid_list)
        self.optimizer = BO(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        self.utility = UtilityFunction(**self.utility_params)

        score_best = AdvCost(float("inf"))
        q_best = None
        for i in range(self.num_initial_samples):
            q = random.choice(Q)
            score = self.perturb_agents(scene, q, cav_id_list, rsu_id_list)
            q_bo = self.convert_q_to_BO_format(q)
            # Ensure unique observations
            if self.optimizer._space._as_array(
                    q_bo) not in self.optimizer._space:
                self.optimizer.register(params=q_bo, target=score.adv_loss)

            if score < score_best:
                score_best = score
                q_best = q
        return score_best, q_best

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
        # score_first: The perception score for the first random
        # selected cav combination.
        # score_cav_selection: The perception score for the optimal cav selection.
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
        print(f"Q size: {len(Q)}")
        bo_initial_score, q_best = self.init_model(scene, Q, vid_list,
                                                   cav_id_list, rsu_id_list)
        # score_min: the minimum perception score for the final perturbed scene.
        score_min = min(bo_initial_score, score_cav_selection)
        q_best = q_best if bo_initial_score < score_cav_selection else None
        for i in range(self.max_iter):
            next_q_bo = self.optimizer.suggest(self.utility)
            next_q = self.convert_q_bo_to_carla_format(next_q_bo)
            next_q = self.project_q_to_Q_if_needed_else_add_q_to_Q(scene,
                                                                   next_q, Q)
            next_q_bo = self.convert_q_to_BO_format(next_q)
            # Ensure unique observations
            if self.optimizer._space._as_array(
                    next_q_bo) in self.optimizer._space:
                continue
            score = self.perturb_agents(scene, next_q, cav_id_list,
                                        rsu_id_list)

            q_best = next_q if score < score_min else q_best
            score_min = min(score, score_min)
            self.optimizer.register(params=next_q_bo,
                                    target=score.adv_loss)

        self.register_agent_score_drop(q_best,
                                       bo_initial_score.adv_loss - score_min.adv_loss)

        if vis_flag:
            self.get_adversarial_cost(scene, cav_id_list, rsu_id_list,
                                      vis_flag=vis_flag,
                                      vis_save_path="./adversarial_position.png")
        out = {
            "score_first": score_first,
            "score_cav_selection": score_cav_selection,
            "bo_initial_score": bo_initial_score,
            "score_best": score_min,
            "cav_id_list": cav_id_list,
            "rsu_id_list": rsu_id_list,
            "q_best": q_best,
            "weight_dict": weight_dict
        }
        return out
