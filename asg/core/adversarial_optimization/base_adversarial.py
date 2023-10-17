import copy
import os.path
import random
from collections import defaultdict
from itertools import combinations, product

import carla
import numpy as np

import opencood.utils.common_utils as common_utils
from opencood.data_utils.datasets import build_dataset
from opencood.model_wrapper import ModelWrapper
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization.vis_utils import visualize_lidar_ego
from asg.data_utils.scene_datasets.data_dumping_utils import dump_data_opencda, \
    dump_agent_weights, load_agent_weight
from asg.utils.adversarial_utils import AdvCost
from asg.utils.geometry_utils import sort_agent_order_by_occlusion_level
from asg.utils.adversarial_utils import softmax



def register_agent_score_drop(f):
    def wrapper(*args, **kwargs):
        args[0].init_agent_score_drop_registration()
        res = f(*args, **kwargs)
        args[0].clear_agent_score_drop_registration()
        return res

    return wrapper


class AdversarialLogger:
    def __init__(self):
        pass

    def init_agent_score_drop_registration(self):
        self.agent_score_drop_pair = defaultdict(int)
        self.agent_delta_pose_pair = defaultdict(
            lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def register_agent_score_drop(self, q, score_drop):
        if q is None:
            return
        if score_drop > 0:
            for vid in q.keys():
                self.agent_score_drop_pair[vid] += score_drop / len(q.keys())
                self.agent_delta_pose_pair[vid] += np.array(
                    [q[vid].location.x, q[vid].location.y, q[vid].location.z,
                     q[vid].rotation.pitch, q[vid].rotation.yaw,
                     q[vid].rotation.roll])

    def clear_agent_score_drop_registration(self):
        self.agent_score_drop_pair = None
        self.agent_delta_pose_pair = None


class BaseAdversarial(AdversarialLogger):
    """
    A class for base adversarial algorithms. All the adversarial algorithms
    should inherit this class.
    """

    def __init__(self, opt, config):
        self.opt = opt
        self.use_original_cavs = opt.use_original_cavs
        self.discard_cav_selection = opt.discard_cav_selection
        self.agent_weight_dir = opt.agent_weight_dir

        if self.use_original_cavs:
            assert self.opt.load_opencda_format_data, "To use original CAVS" \
                                                      "users must use OpenCDA" \
                                                      "format data."
        self.config = config
        self.num_cavs = config["num_cavs"]
        self.num_infra = config["num_infra"]
        self.N_sample = config["N_sample"]
        self.N_min = config["N_min"]
        self.num_cav_trials = config["num_cav_trials"]
        self.k = config["k"]
        self.N = config["N"]
        self.interval = config["interval"]

        # only perturb the yaw
        self.perturb_angle_range = config["angle_range"]

        self.model_params = load_yaml(None, opt)

        self.model = ModelWrapper(opt)
        self.opencda_dataset = build_dataset(self.model_params,
                                             train=False,
                                             visualize=True)

        self.x_range = (-((self.N - 1) // 2) * self.interval,
                        ((self.N - 1) // 2) * self.interval)
        self.y_range = (-((self.N - 1) // 2) * self.interval,
                        ((self.N - 1) // 2) * self.interval)

    def get_bounds(self, vid_list):
        # todo need to refactor

        pbounds_min = np.tile(np.array([self.x_range[0],
                                        self.y_range[0],
                                        self.perturb_angle_range[0]]),
                              (len(vid_list), 1))
        pbounds_max = np.tile(np.array([self.x_range[1],
                                        self.y_range[1],
                                        self.perturb_angle_range[1]]),
                              (len(vid_list), 1))
        pbounds = {"min": pbounds_min, "max": pbounds_max}
        return pbounds

    def select_rsu(self, i, scene):
        # for simplicity, the below code assumes constent infra/CAV ratio
        # todo decide the type from collected data instead
        # v2v vs v2I
        self.num_cavs = self.config["num_cavs"]
        assert self.opt.v2x in ["v2v", "v2i", "v2x"]
        if self.opt.v2x == "v2x":
            flag = i % 3

        if self.opt.v2x == "v2v" or (self.opt.v2x == "v2x" and flag == 0):
            vid_list = \
                scene.get_vehicle_id_list_with_type_within_spawn_range("car")
            vid_list = sorted(vid_list)
            rsu_id_list = [vid_list[0]]
            assert self.num_cavs >= 0
        elif self.opt.v2x == "v2i" or (self.opt.v2x == "v2x" and flag != 0):
            rsu_id_list = [-1]
        else:
            raise ValueError(f"opt.v2x must in ['v2v', 'v2i'] "
                             f"but received {self.opt.v2x}")
        return rsu_id_list


    def select_adversarial_cavs_attn(self, rsu_id_list, scene, vis_flag=False):
        """
        Find adversarial CAV combination by attention weights.
        If self.discard_cav_selection==True, directly output with the initial
        CAV selection.
        Args:
            rsu_id_list: list
                RSU id list
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.

        Returns:
            best_cav_id_list: list
                List of IDs for the best cav combination.
            score_best: float
                Best associated perception score.
            score_first: float
                The perception score for the first random selected cav combination.

        """
        vid_list = \
            scene.get_vehicle_id_list_with_type_within_spawn_range("car")
        vid_list = sorted(vid_list)
        vid_list = [x for x in vid_list if x not in rsu_id_list]
        truncated_vid_list = vid_list

        # The initial cav combination
        if not self.use_original_cavs:
            num_cavs = min(self.num_cavs, len(vid_list))
            cav_id_list = list(
                next(combinations(truncated_vid_list, num_cavs)))
        # If using the original CAV as the initialization, then set the num_cavs
        # to the original CAV number instead of the number stored in the
        # adversarial optimization config file.
        else:
            cav_id_list = sorted(scene.get_original_cav_id_list())
            num_cavs = len(cav_id_list)
        score_first = score_best = self.get_adversarial_cost(scene,
                                                             cav_id_list,
                                                             rsu_id_list)
        best_cav_id_list = cav_id_list
        if vis_flag:
            self.get_adversarial_cost(scene, cav_id_list, rsu_id_list,
                                      vis_flag=vis_flag,
                                      vis_save_path="./initial.png")
        scenario_name, timestamp = scene.get_scenario_name_and_timstamp()
        weight_dict = {}
        if not self.discard_cav_selection:
            if len(self.agent_weight_dir) == 0:
                weight = self.get_attn(scene, truncated_vid_list, rsu_id_list)
                for i, vid in enumerate(truncated_vid_list):
                    weight_dict[vid] = weight[i + 1]
            else:
                weight_dict = load_agent_weight(self.agent_weight_dir,
                                                scenario_name, timestamp)
            print(weight_dict)
            cav_id_list_comb = list(combinations(truncated_vid_list, num_cavs))
            num_cav_trials = min(len(cav_id_list_comb), self.num_cav_trials)
            weight_list = []
            for pair in cav_id_list_comb:
                weight_list.append(sum(1 / (weight_dict[itm]) for itm in pair))
            weight_list = np.array(weight_list)
            # weighted sample without replacement
            index = list(range(len(cav_id_list_comb)))
            # convert weight to probability
            tau = 0.03
            p = softmax(weight_list / tau)
            while (p > 0).sum() < num_cav_trials:
                tau += 0.1
                p = softmax(weight_list / tau)
            print("p:", p[:5], "tau: ", tau)
            sampled_cav_id_list_index = np.random.choice(index, \
                                                         size=num_cav_trials,
                                                         replace=False,
                                                         p=p)
            sampled_cav_id_list_comb = [cav_id_list_comb[idx] for idx in
                                        sampled_cav_id_list_index]

            for i, cav_id_list in enumerate(sampled_cav_id_list_comb):
                cav_id_list = sorted(list(cav_id_list))
                score = self.get_adversarial_cost(scene, cav_id_list,
                                                  rsu_id_list)
                if score < score_best:
                    score_best = score
                    best_cav_id_list = cav_id_list

        return best_cav_id_list, score_best, score_first, weight_dict
    def perturb_agents(self, scene, q, cav_id_list, rsu_id_list,
                       convert_back=True):
        """
        Perturb each agent.
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            q: dict
                Dictionary of {vid: carla.Transform } with vid in the list of
                selected vehicle IDs to perturb.
            cav_id_list: list
                List of selected CAV IDs.
            rsu_id_list: list
                List of RSU IDs.
            convert_back: boolean
                True if converting back to the pre-perturbed scenes for search
                roll out.

        Returns:
            previous_score: float
                Perception score for the perturbed scene.

        """
        original_transform_dict = {}
        for vid in q.keys():
            original_transform = scene.agent_original_poses[vid]
            original_rotation = original_transform.rotation
            new_transform = carla.Transform(
                location=original_transform.location + q[vid].location,
                rotation=carla.Rotation(
                    pitch=original_rotation.pitch + q[vid].rotation.pitch,
                    yaw=original_rotation.yaw + q[vid].rotation.yaw,
                    roll=original_rotation.roll + q[vid].rotation.roll))

            scene.set_agent_transform(vid, new_transform)

            original_transform_dict.update({vid: original_transform})
        scene.tick()

        new_score = self.get_adversarial_cost(scene, cav_id_list,
                                              rsu_id_list)
        if convert_back:
            for vid in scene.agent_original_poses.keys():
                scene.set_agent_transform(vid, scene.agent_original_poses[vid])
            scene.tick()

        return new_score

    def get_vid_list_and_feasible_set(self, scene, rsu_id_list, cav_id_list):
        """

        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            N_min: int
                Required minimum number size of feasible set Q.
                If the feasible set size is less than this number, try another
                perturbation agent selection.

        Returns:
            vid_list: list
                List of vehicle IDs to perturb
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all
                valid perturbations.

        """
        # List all possible vid_list combinations with length k
        all_vid_list = \
            scene.get_vehicle_id_list_within_spawn_range()
        all_vid_list = [x for x in all_vid_list if x not in rsu_id_list]
        # {id: Vehicle/RSU_LiDAR object}
        veh_dict = {k: v1 if k >= 0 else v2 for k, (v1, v2) in
                    scene.agent_pair_id_map.items()}

        # sorted all vid combinations according to occlusion level
        occ_score_map = sort_agent_order_by_occlusion_level(
            self.k, veh_dict,
            all_vid_list,
            rsu_id_list,
            cav_id_list)

        sorted_vid_list = sorted(all_vid_list,
                                 key=lambda vid: -occ_score_map[vid])
        # prune single vehicles that has too little search space
        all_vid_list_pruned = copy.deepcopy(sorted_vid_list)
        for vid in sorted_vid_list:
            Q = self.generate_random_perturbation([vid], scene)
            print(vid, len(Q))
            if len(Q) < self.N_min / 2:
                all_vid_list_pruned.remove(vid)
            else:
                break

        all_vid_list = list(combinations(all_vid_list_pruned, self.k))
        # sort in decreasing order
        sorted_all_vid_list = sorted(all_vid_list, \
                                     key=lambda vids: \
                                         -sum(occ_score_map[vid] for vid in
                                              vids))

        for vid_list in sorted_all_vid_list:
            vid_list = sorted(vid_list)
            Q = self.generate_random_perturbation(vid_list, scene)
            print("Q: ", len(Q))
            if len(Q) > self.N_min:
                break
        else:
            print(
                f"Can't find any feasible set Q with size at least {self.N_min}")
            return [], []
        return vid_list, Q

    def get_random_vid_list_with_no_empty_feasible_set(self, scene):
        """

        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.

        Returns:
            vid_list: list
                List of vehicle IDs to perturb
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all
                valid perturbations.

        """
        # List all possible vid_list combinations with length k
        all_vid_list = \
            scene.get_vehicle_id_list_within_spawn_range()
        all_vid_list = list(combinations(all_vid_list, self.k))
        random.shuffle(all_vid_list)
        # Ensure the size of feasible set Q is larger than a threshold
        for vid_list in all_vid_list:
            vid_list = sorted(vid_list)
            Q = self.generate_random_perturbation(vid_list, scene)
            print("Q: ", len(Q))
            if len(Q) > self.N_min:
                break
        else:
            print(
                f"Can't find any feasible set Q with size at least {self.N_min}")
            return [], []
        return vid_list, Q

    def generate_uniform_perturbation(self, vid_list, scene):
        """
        Generate the perturbations for vehicle vid.
        Args:
            vid_list: list
                List of vehicle IDs for agents to perturb.
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.

        Returns:
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all
                valid perturbations.

        """
        assert len(vid_list) > 0
        N_angle = int(
            (self.perturb_angle_range[1] - self.perturb_angle_range[0]) /
            self.perturb_angle_range[2]) + 1
        Q_angle = list(np.linspace(self.perturb_angle_range[0],
                                   self.perturb_angle_range[1], num=N_angle))
        # N^2-1
        Q_loc = [[self.interval * i, self.interval * j] for i in
                 range(-(self.N - 1) // 2, (self.N + 1) // 2) for j in
                 range(-(self.N - 1) // 2, (self.N + 1) // 2) if
                 i != 0 or j != 0]

        # (N^2-1) * N_angle
        Q_single = [(dx, dy, dtheta) for (dx, dy) in Q_loc for dtheta in
                    Q_angle]

        # sample N_sample from all the possible combinations to reduce the
        # search space
        # can lead to large memory
        Q_product = list(product(*([Q_single] * len(vid_list))))

        if self.N_sample < len(Q_product):
            Q_sample = random.sample(Q_product, self.N_sample)
        else:
            Q_sample = Q_product

        Q = []
        for q_batch in Q_sample:
            q_dict = {}
            for i, vid in enumerate(vid_list):
                q_loc = carla.Location(q_batch[i][0], q_batch[i][1], 0)
                # only perturb yaw angle
                q_rot = carla.Rotation(0, q_batch[i][2], 0)
                q = carla.Transform(location=q_loc, rotation=q_rot)
                q_dict.update({
                    vid: q
                })

            valid_perturbation = self.valid_position_for_multi_agents(scene,
                                                                      q_dict)
            if valid_perturbation:
                Q.append(q_dict)

        return Q

    def generate_random_perturbation(self, vid_list, scene, method="uniform"):
        """
        Generate the perturbations for vehicle vid.
        Args:
            vid_list: list
                List of vehicle IDs for agents to perturb.
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            method: str
                Sampling method: "uniform" sampling or "grid" based sampling.

        Returns:
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all
                valid perturbations.

        """
        assert len(vid_list) > 0 and method in ["uniform", "grid"]
        if method == "uniform":
            pbounds = self.get_bounds(vid_list)
            Q_sample = np.random.uniform(pbounds["min"], high=pbounds["max"],
                                         size=(
                                             self.N_sample, len(vid_list), 3))
        else:
            assert NotImplementedError

        Q = []
        for q_batch in Q_sample:
            q_dict = {}
            for i, vid in enumerate(vid_list):
                q_loc = carla.Location(q_batch[i][0], q_batch[i][1], 0)
                # only perturb yaw angle
                q_rot = carla.Rotation(0, q_batch[i][2], 0)
                q = carla.Transform(location=q_loc, rotation=q_rot)
                q_dict.update({
                    vid: q
                })

            valid_perturbation = self.valid_position_for_multi_agents(scene,
                                                                      q_dict)
            if valid_perturbation:
                Q.append(q_dict)

        return Q

    def valid_position_for_multi_agents(self, scene, q_dict):
        """
        Check if the perturbation q_dict is valid
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            q_dict: dict
                Multi-agent perturbation with format {vid: carla.Transform}

        Returns:
            valid_perturbation: boolean
                True if the perturbation is valid

        """
        valid_perturbation = True
        other_agent_transforms = {}
        vid_list = sorted(q_dict.keys())
        for i, vid in enumerate(vid_list):
            agent = scene.get_agent(vid)
            original_transform = agent.get_transform()

            q_loc = q_dict[vid].location
            q_rot = q_dict[vid].rotation

            new_transform = carla.Transform(
                location=original_transform.location + q_loc,
                rotation=carla.Rotation(
                    pitch=original_transform.rotation.pitch + q_rot.pitch,
                    yaw=original_transform.rotation.yaw + q_rot.yaw,
                    roll=original_transform.rotation.roll + q_rot.roll))

            # All agents in the vid_list that are perturbed so far
            other_agent_transforms.update({
                vid: new_transform
            })

            if not self.valid_position_for_single_agent(vid, new_transform,
                                                        scene,
                                                        other_agent_transforms=other_agent_transforms):
                valid_perturbation = False
                break
        return valid_perturbation

    def project_q_to_Q_if_needed_else_add_q_to_Q(self, scene, q, Q):
        """
        Project q to Q if q is invalid position. Else return q directly and
        update Q with q.
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            q: dict
                Multi-agent perturbation with format {vid: carla.Transform}
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all
                valid perturbations.

        Returns:
            q_proj: dict
                Valid multi-agent perturbation (projected or intact) with
                format {vid: carla.Transform}

        """
        # If the perturbation q is valid, then update Q with q and return q
        # directly
        if self.valid_position_for_multi_agents(scene, q):
            Q.append(q)
            return q
        # If the perturbation q is invalid, then project q to Q
        vid_list = sorted(q.keys())
        Q_np = self.convert_Q_to_numpy(Q)
        q_proj = self.project_q_to_Q(q, Q_np, vid_list)
        return q_proj

    def convert_Q_to_numpy(self, Q):
        """
        Convert list of CARLA perturbation Q to numpy format.
        Args:
            Q: list
                List: [{vid: carla.Transform }] for vid in vid_list and all
                valid perturbations.

        Returns:
            Q_np: np.ndarray
                Numpy array format of feasible set Q with shape (len(Q), 3).

        """
        # (len(Q), len(vid_list), 3) where the last dimension is (dx,dy,dyaw)
        Q_np = np.stack(self.convert_q_to_numpy(q) for q in Q)
        return Q_np

    def convert_q_to_numpy(self, q):
        """
        Convert CARLA perturbation q to numpy format.
        Args:
            q: dict
                Dict of CARLA perturbation with format {vid: carla.Transform}

        Returns:
            q_np: np.ndarray
                Numpy array format of q with shape (len(vid_list), 3)

        """
        vid_list = sorted(q.keys())
        q_np = [[q[vid].location.x, q[vid].location.y, q[vid].rotation.yaw] for
                vid in vid_list]
        # (len(vid_list), 3)
        q_np = np.array(q_np)
        return q_np

    def convert_q_np_to_q(self, q_np, vid_list):
        """
        Convert q_np back to CARLA perturbation q format.
        Args:
            q_np: np.ndarray
                Numpy array format of q with shape (len(vid_list), 3)
            vid_list: list
                Sorted list of vehicle IDs that is perturbed.

        Returns:
            q: dict
                Dict of CARLA perturbation with format
                {vid: carla.Transform}

        """
        vid_list = sorted(vid_list)
        q = {}
        for i, vid in enumerate(vid_list):
            q_loc = carla.Location(q_np[i][0], q_np[i][1])
            q_rot = carla.Rotation(yaw=q_np[i][2])
            q.update({vid: carla.Transform(location=q_loc, rotation=q_rot)})
        return q

    def project_q_to_Q(self, q, Q_np, vid_list):
        """
        Project q to the feasible set Q.
        Args:
            q: dict
                Dict of CARLA perturbation with format
                {vid: carla.Transform}
            Q_np: np.ndarray
                Numpy array format of feasible set Q with shape (len(Q), 3).
            vid_list: list
                Sorted list of vehicle IDs that is perturbed.

        Returns:
            q_proj: dict
                Valid projected perturbation (projected or intact) with format
                {vid: carla.Transform}

        """
        q_np = self.convert_q_to_numpy(q)
        q_np = np.expand_dims(q_np, 0)
        dist = np.linalg.norm(Q_np - q_np, axis=-1).sum(-1)
        q_proj_np = Q_np[dist.argmin(), :, :]
        q_proj = {vid: carla.Transform(
            location=carla.Location(q_proj_np[i, 0], q_proj_np[i, 1]),
            rotation=carla.Rotation(yaw=q_proj_np[i, 2])) for i, vid in
            enumerate(vid_list)}
        return q_proj

    def get_adversarial_cost(self, scene, cav_id_list, rsu_id_list,
                             vis_flag=False, vis_save_path=''):
        """
        Get adversarial cost for the current scene
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            cav_id_list: list
                List of selected CAV IDs.
            rsu_id_list: list
                List of RSU IDs.
            vis_flag: boolean
                True if visualize the lidar, gt and predictions.
            vis_save_path: str
                Path for saving the visualization.

        Returns:
            adv_cost: AdvCost
                The adversarial cost of the current scene.

        """

        weight = {0.3: 1, 0.5: 0.8, 0.7: 0.5}
        observation = scene.get_observation(cav_id_list, rsu_id_list)
        gt = scene.get_gt_bbx()
        ap_dict, result_stats = self.get_ap_score(observation, gt,
                                                  vis_flag=vis_flag,
                                                  vis_save_path=vis_save_path)
        score = sum(weight[k] * ap_dict[k] for k in weight.keys())
        adv_cost = AdvCost(adv_loss=score,
                           ap_dict=ap_dict,
                           result_stats=result_stats,
                           cav_id_list=cav_id_list,
                           rsu_id_list=rsu_id_list,
                           observation=observation,
                           gt=gt)
        return adv_cost

    def get_attn(self, scene, cav_id_list, rsu_id_list,
                 vis_flag=False, vis_save_path=''):
        """
        Get attention weight for the current scene
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            cav_id_list: list
                List of selected CAV IDs.
            rsu_id_list: list
                List of RSU IDs.
            vis_flag: boolean
                True if visualize the lidar, gt and predictions.
            vis_save_path: str
                Path for saving the visualization.

        Returns:
            adv_cost: AdvCost
                The adversarial cost of the current scene.

        """

        observation = scene.get_observation(cav_id_list, rsu_id_list)
        gt = scene.get_gt_bbx()

        # from opencood.visualization import vis_utils
        processed_data = self.opencda_dataset.data_reformat(observation, gt)
        tensor_data = self.opencda_dataset.collate_batch_test([processed_data])

        # perform inference
        pred_box_tensor, pred_score, gt_box_tensor, attn = \
            self.model.get_attention_weight(tensor_data, self.opencda_dataset)
        # evaluate
        ap_dict, result_stats = self.model.evaluate(pred_box_tensor,
                                                    pred_score,
                                                    gt_box_tensor)
        pos_equal_one = tensor_data['ego']["label_dict"]["pos_equal_one"][
            0, ...].to(attn.device)
        if len(attn.shape) > 1:
            weight = (pos_equal_one.sum(-1) > 0).unsqueeze(0) * attn[0, ...]
            weight = weight.sum(-1).sum(-1).detach().cpu().numpy()
        else:
            weight = attn.detach().cpu().numpy()
        return weight

    def get_ap_score(self, observation, gt_info, vis_flag=False,
                     vis_save_path=''):
        # from opencood.visualization import vis_utils
        processed_data = self.opencda_dataset.data_reformat(observation,
                                                            gt_info)

        tensor_data = self.opencda_dataset.collate_batch_test([processed_data])

        # perform inference
        pred_box_tensor, pred_score, gt_box_tensor = \
            self.model.inference(tensor_data, self.opencda_dataset)
        # evaluate
        ap_dict, result_stats = self.model.evaluate(pred_box_tensor,
                                                    pred_score,
                                                    gt_box_tensor)

        # visualize
        if vis_flag:
            pair = {k: v / max(self.agent_score_drop_pair.values()) for k, v in
                    self.agent_score_drop_pair.items()}
            color_dict = {k: (1 - v, 1 - v, 1 - v) for k, v in pair.items()}
            if len(pair):
                max_key = max(pair, key=pair.get)
                color_dict[max_key] = (1, 0, 0)
            self.model.visualize(pred_box_tensor,
                                 gt_box_tensor,
                                 tensor_data['ego']['origin_lidar'][0],
                                 self.opencda_dataset,
                                 show_vis=False,
                                 vis_save_path=vis_save_path,
                                 color_dict=color_dict)
        return ap_dict, result_stats

    def visualize_lidar(self, observation, gt_info):
        processed_data = self.opencda_dataset.data_reformat(observation,
                                                            gt_info)
        tensor_data = self.opencda_dataset.collate_batch_test([processed_data])
        self.model.visualize(None,
                             None,
                             tensor_data['ego']['origin_lidar'][0],
                             self.opencda_dataset)

    def valid_position_for_single_agent(self, vid, new_transform, scene,
                                        other_agent_transforms={}):
        """
        Check if the new transform is a valid position
        Args:
            vid: int
                Vehicle id for agent to perturb.
            new_transform: carla.Transform
                New position for vehicle with id vid.
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            other_agent_transforms: dict
                Dictionary of {other_agent_vid: new CARLA transform positions}. Default is None, which means no other agents are concurrently perturbed.

        Returns:
            flag: boolean
                True if the new_transform is a valid position.
        """
        if not self.check_if_pose_in_range(vid, new_transform, scene):
            return False

        # The empty dictionary is a dangerous default value for as discussed
        # in https://stackoverflow.com/questions/26320899/why-is-the-empty
        # -dictionary-a-dangerous-default-value-in-python Thus we use None
        # as the default value and convert the default None to dict for code
        # simplicity. Convert the default None to default dict
        other_agent_transforms = {} if other_agent_transforms is None \
            else other_agent_transforms

        for i in scene.get_vehicle_id_list():
            if i == vid:
                continue

            new_transform2 = other_agent_transforms.get(i, None)

            if self.check_if_two_vehicles_collide(vid, i, scene,
                                                  transform1=new_transform,
                                                  transform2=new_transform2):
                return False

        return True

    def check_if_pose_in_range(self, vid, new_transform, scene):
        """
        Check if the pose of agent vid is within the spawn range.
        Args:
            vid: int
                Vehicle ID
            new_transform: carla.Transform
                New position for vehicle vid.
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.

        Returns:
            True if the entire bbx of the perturbed agent is within the range.
            Otherwise False.

        """
        veh_bbx = scene.get_extended_bbx_in_world(vid, transform=new_transform)
        veh_bbx_xy = veh_bbx[:, :2]
        rsu_center_pose = np.asarray(scene.rsu_center_pose).reshape(1, -1)
        # To make sure the bbx is within the range,
        # the range is shrined a little bit compared with original spanw range.
        spawn_range = scene.spawn_range - 5
        dist = np.linalg.norm(veh_bbx_xy - rsu_center_pose, axis=-1)
        return (dist < spawn_range).all()

    def check_if_two_vehicles_collide(self, vid1, vid2, scene, transform1=None,
                                      transform2=None):
        """
        Check if two vehicles will collide with each other given the new
         transform (if specified).
        Args:
            vid1: int
                Vehicle1 ID
            vid2: int
                Vehicle2 ID
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            transform1: carla.Transform
                New position for vehicle with id vid1. If None,
                use the original vehicle's transform
            transform2: carla.Transform
                New position for vehicle with id vid2. If None,
                 use the original vehicle's transform.

        Returns:
            flag: boolean
                Return True if two vehicles will collide with each other.
                 Otherwise, return False.

        """
        assert vid1 != vid2, \
            "Vid1 can't equal to vid2, otherwise they are the same vehicle"
        veh1_bbx = scene.get_extended_bbx_in_world(vid1, transform=transform1)
        veh2_bbx = scene.get_extended_bbx_in_world(vid2, transform=transform2)
        bbx_stacked = np.concatenate([np.expand_dims(veh1_bbx, axis=(0)),
                                      np.expand_dims(veh2_bbx, axis=(0))],
                                     axis=0)
        polygons = common_utils.convert_format(bbx_stacked)
        veh1_polygon = polygons[0]
        veh2_polygon = polygons[1]
        iou = common_utils.compute_iou(veh1_polygon, [veh2_polygon])[0]
        flag = iou != 0
        return flag

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
        pass

    def dump_data(self, saved_path, scene, out):
        """
        Dump the current scene's data to OpenCDA format.
        Args:
            scene: asg.core.scene_manager.SceneManager
                SceneManager object that can interact with CARLA.
            out: dict
                Adversarial attack model's output.

        """
        cav_id_list = out["cav_id_list"]
        rsu_id_list = out["rsu_id_list"]
        score_best = out["score_best"]
        weight_dict = out["weight_dict"]
        # observation for cav&infra with lidar_np, lidar_pose,
        # indexed by id in cav_id_list or rsu_id_list
        observation = score_best.observation
        # GT for all vehicles
        gt = score_best.gt

        scenario_name, timestamp = scene.get_scenario_name_and_timstamp()
        scenario_folder = os.path.join(saved_path,
                                       "generated_scenes",
                                       scenario_name)

        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)
        scene.current_sensor_manager.dump_config(scene.current_scene_meta,
                                                 config_name="data_protocol.yaml",
                                                 save_path=scenario_folder)

        # dump data to folder
        dump_data_opencda(saved_path,
                          scenario_name,
                          timestamp,
                          rsu_id_list + cav_id_list,
                          observation,
                          gt)
        # dump rsu_id_list and cav_id_list
        if len(weight_dict):
            dump_agent_weights(saved_path,
                               scenario_name,
                               timestamp,
                               weight_dict)
