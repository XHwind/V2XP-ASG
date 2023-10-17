from asg.core.adversarial_optimization.base_adversarial import *


class RandomSearch(BaseAdversarial):
    """
    A class for random search algorithm.

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
        super(RandomSearch, self).__init__(opt, config)
        self.max_iter = config["max_iter"]
        self.score_noise_margin = 1e-2

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
        scenario_name, timestamp = scene.get_scenario_name_and_timstamp()

        rsu_id_list = self.select_rsu(i, scene)
        cav_id_list, score_cav_selection, score_first, weight_dict = \
            self.select_adversarial_cavs_attn(rsu_id_list,
                                              scene,
                                              vis_flag=vis_flag)

        print(f"rsu_id_list: {rsu_id_list} cav_id_list: {cav_id_list}")
        score_min = score_cav_selection
        q_best = None
        if vis_flag and not self.discard_cav_selection:
            vis_save_path = "_".join(
                [scenario_name, timestamp, "cav_selection.png"])
            score = self.get_adversarial_cost(scene, cav_id_list,
                                              rsu_id_list, vis_flag=vis_flag,
                                              vis_save_path=vis_save_path)
        if self.max_iter > 0:
            vid_list, Q = self.get_vid_list_and_feasible_set(scene,
                                                             rsu_id_list,
                                                             cav_id_list)

        for i in range(self.max_iter):
            # algorithm to get the next position
            q = random.choice(Q)
            score = self.perturb_agents(scene, q, cav_id_list, rsu_id_list)
            q_best = q if score < score_min else q_best
            score_min = min(score, score_min)

        self.register_agent_score_drop(q_best,
                                       score_cav_selection.adv_loss - score_min.adv_loss)

        if vis_flag and q_best:
            self.perturb_agents(scene, q_best, cav_id_list, rsu_id_list,
                                convert_back=False)
            vis_save_path = "_".join([scenario_name, timestamp,
                                      "adversarial_position.png"])
            self.get_adversarial_cost(scene, cav_id_list, rsu_id_list,
                                      vis_flag=vis_flag,
                                      vis_save_path=vis_save_path)

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
