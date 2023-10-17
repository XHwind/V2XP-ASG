import os
import copy

from datetime import datetime

import numpy as np

import opencood.utils.eval_utils as eval_utils
from asg.hypes_yaml.yaml_utils import save_yaml, load_yaml


class AdvCost:
    """
    Class for encapsulating the adversarial cost, APs and internal statistics
    for calculating APs. This class support various comparison operators by
    using the value of adv_loss. Other attributes like ap_dict and result_stats
    are used for calculating overall APs.
    Attributes:
        self.adv_loss: AdvCost
            Adversarial cost for the current scene
        self.ap_dict: dict
            {iou threshold: AP score}
        self.result_stats: dict
                {iou threshold: {"fp":list, "tp":list, "gt":int}}}

    """

    def __init__(self,
                 adv_loss=float("inf"),
                 ap_dict=None,
                 result_stats=None,
                 cav_id_list=None,
                 rsu_id_list=None,
                 observation=None,
                 gt=None):
        self.adv_loss = adv_loss
        self.ap_dict = ap_dict if ap_dict is not None else {}
        self.result_stats = result_stats if result_stats is not None else {}
        self.cav_id_list = cav_id_list if cav_id_list is not None else []
        self.rsu_id_list = rsu_id_list if rsu_id_list is not None else []
        self.observation = observation
        self.gt = gt

    def __eq__(self, other):
        return isinstance(other, AdvCost) and \
               self.adv_loss == other.adv_loss

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.adv_loss < other.adv_loss

    def __gt__(self, other):
        return self.adv_loss > other.adv_loss

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other

    def __str__(self):
        ap_msg = "{"
        for i, (k, v) in enumerate(self.ap_dict.items()):
            ap_msg += f"{k}: {v:.4f}"
            ap_msg += ", " if i < len(self.ap_dict.items()) - 1 else ""
        ap_msg += "}"
        msg = f"{{adv_Loss: {self.adv_loss:.4f}, ap_dict: {ap_msg}}}"
        return msg


class AdvCostAggregator:
    """
    Aggregate perception statistics across the scenes.
    """

    def __init__(self, cum_adv_loss=0.0, cum_N=0.0, all_result_stat=None):
        self.adv_cost_list = []
        self.iou_threshs = [0.3, 0.5, 0.7]
        self.cum_adv_loss = cum_adv_loss
        self.cum_N = cum_N

        if all_result_stat is None:
            self.all_result_stat = {iou_thresh: {'tp': [], 'fp': [], 'gt': 0}
                                    for iou_thresh in self.iou_threshs}
        else:
            self.all_result_stat = all_result_stat

    def add(self, adv_cost):
        """
        Update  the aggregator with the current adversarial scene's cost.
        Args:
            adv_cost: AdvCost
                The adversarial cost of the current scene.

        """
        assert isinstance(adv_cost, AdvCost) and adv_cost != AdvCost(float(
            "inf")), f"adv_cost should be AdvCost object and should not have cost of inf"

        self.adv_cost_list.append(adv_cost)
        self.update_result_stats(adv_cost.result_stats)
        self.update_adv_loss(adv_cost.adv_loss)

    def update_adv_loss(self, adv_loss):
        self.cum_adv_loss += adv_loss
        self.cum_N += 1

    def get_avg_adv_cost(self):
        """
        Get overall adversarial cost, APs etc.
        Returns:
            adv_cost: AdvCost
                The overall adversarial cost.
        """
        avg_adv_loss = self.cum_adv_loss / self.cum_N
        APs = self.get_APs()
        adv_cost = AdvCost(adv_loss=avg_adv_loss, ap_dict=APs,
                           result_stats=copy.deepcopy(self.all_result_stat))
        return adv_cost

    def get_APs(self):
        APs = {}
        for iou_thresh in self.iou_threshs:
            ap, _, _ = eval_utils.calculate_ap(
                copy.deepcopy(self.all_result_stat),
                iou_thresh)
            APs[iou_thresh] = ap
        return APs

    def get_avg_APs(self):
        """
        Calculate the average APs for debugging AP calculation.
        Returns:
            APs: dict
                {iou threshold: AP}

        """
        APs = {iou_thresh: [] for iou_thresh in self.iou_threshs}

        for adv_cost in self.adv_cost_list:
            aps = adv_cost.ap_dict
            for iou_thresh in self.iou_threshs:
                APs[iou_thresh].append(aps[iou_thresh])

        for iou_thresh in self.iou_threshs:
            APs[iou_thresh] = np.mean(APs[iou_thresh])
        return APs

    def update_result_stats(self, result_stats):
        """
        Update the all_result_stats with the current scene's result_stats.
        Args:
            result_stats: dict
                {iou threshold: {"fp":list, "tp":list, "gt":int}}}

        """
        for iou_thresh in self.iou_threshs:
            fp = result_stats[iou_thresh]["fp"]
            tp = result_stats[iou_thresh]["tp"]
            gt = result_stats[iou_thresh]["gt"]
            self.all_result_stat[iou_thresh]["fp"] += fp
            self.all_result_stat[iou_thresh]["tp"] += tp
            self.all_result_stat[iou_thresh]["gt"] += gt


def setup_attack(hypes, opt):
    hypes["opt"] = vars(opt)
    model_name = hypes["adversarial"]["core_method"]
    fusion_name = opt.fusion_method
    detection_model_dir = opt.model_dir
    detection_name = detection_model_dir.split("/")[-1]
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    folder_name = "_".join(
        [model_name, fusion_name, detection_name, folder_name])

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        # save the attack configuration yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        save_yaml(hypes, save_name)
    attack_meta_data = {
        "epoch": -1,  # iteration number
        "score_first": {
            "adv_loss": 0.0,  # the sum of adv loss
            "result_stats": {0.3: {'tp': [], 'fp': [], 'gt': 0},
                             0.5: {'tp': [], 'fp': [], 'gt': 0},
                             0.7: {'tp': [], 'fp': [], 'gt': 0}},
            "ap_dict": {0.3: 0.0, 0.5: 0.0, 0.7: 0.0}},
        "score_cav_selection": {
            "adv_loss": 0.0,
            "result_stats": {0.3: {'tp': [], 'fp': [], 'gt': 0},
                             0.5: {'tp': [], 'fp': [], 'gt': 0},
                             0.7: {'tp': [], 'fp': [], 'gt': 0}},
            "ap_dict": {0.3: 0.0, 0.5: 0.0, 0.7: 0.0}},
        "score_best": {
            "adv_loss": 0.0,
            "result_stats": {0.3: {'tp': [], 'fp': [], 'gt': 0},
                             0.5: {'tp': [], 'fp': [], 'gt': 0},
                             0.7: {'tp': [], 'fp': [], 'gt': 0}},
            "ap_dict": {0.3: 0.0, 0.5: 0.0, 0.7: 0.0}},
        "print_message": ""
    }
    # save attack related meta data for resume attack
    save_name = os.path.join(full_path, "attack_meta_data.yaml")
    save_yaml(attack_meta_data, save_name)

    epoch = -1
    score_first_agg = AdvCostAggregator(
        cum_adv_loss=attack_meta_data["score_first"]["adv_loss"],
        cum_N=epoch + 1,
        all_result_stat=attack_meta_data["score_first"]["result_stats"])
    score_cav_selection_agg = AdvCostAggregator(
        cum_adv_loss=attack_meta_data["score_cav_selection"]["adv_loss"],
        cum_N=epoch + 1,
        all_result_stat=attack_meta_data["score_cav_selection"][
            "result_stats"])
    score_best_agg = AdvCostAggregator(
        cum_adv_loss=attack_meta_data["score_best"]["adv_loss"],
        cum_N=epoch + 1,
        all_result_stat=attack_meta_data["score_best"]["result_stats"])

    return full_path, attack_meta_data, epoch, \
           score_first_agg, score_cav_selection_agg, score_best_agg


def resume_attack(resume_folder):
    save_name = os.path.join(resume_folder, "attack_meta_data.yaml")
    attack_meta_data = load_yaml(save_name)
    epoch = int(attack_meta_data["epoch"])
    score_first_agg = AdvCostAggregator(
        cum_adv_loss=attack_meta_data["score_first"]["adv_loss"],
        cum_N=epoch + 1,
        all_result_stat=attack_meta_data["score_first"]["result_stats"])
    score_cav_selection_agg = AdvCostAggregator(
        cum_adv_loss=attack_meta_data["score_cav_selection"]["adv_loss"],
        cum_N=epoch + 1,
        all_result_stat=attack_meta_data["score_cav_selection"][
            "result_stats"])
    score_best_agg = AdvCostAggregator(
        cum_adv_loss=attack_meta_data["score_best"]["adv_loss"],
        cum_N=epoch + 1,
        all_result_stat=attack_meta_data["score_best"]["result_stats"])
    print(f"Resuming attack from {epoch}")
    print(attack_meta_data["print_message"])
    return attack_meta_data, epoch, score_first_agg, score_cav_selection_agg, score_best_agg


def update_attack_meta(saved_path, attack_meta_data, epoch, score_first_agg,
                       score_cav_selection_agg, score_best_agg, msg):
    attack_meta_data["epoch"] = epoch
    # save result_stats for computing overall APs
    attack_meta_data["score_first"]["result_stats"] = \
        score_first_agg.all_result_stat
    attack_meta_data["score_cav_selection"]["result_stats"] = \
        score_cav_selection_agg.all_result_stat
    attack_meta_data["score_best"]["result_stats"] = \
        score_best_agg.all_result_stat
    # save adversarial loss
    attack_meta_data["score_first"]["adv_loss"] = score_first_agg.cum_adv_loss
    attack_meta_data["score_cav_selection"]["adv_loss"] = \
        score_cav_selection_agg.cum_adv_loss
    attack_meta_data["score_best"]["adv_loss"] = score_best_agg.cum_adv_loss
    # current AP dictionary for viewing only
    attack_meta_data["score_first"]["ap_dict"] = score_first_agg.get_APs()
    attack_meta_data["score_cav_selection"]["ap_dict"] = \
        score_cav_selection_agg.get_APs()
    attack_meta_data["score_best"]["ap_dict"] = score_best_agg.get_APs()
    # print message for debugging purpose
    attack_meta_data["print_message"] += f"\nIteration: {epoch}\n" + msg

    save_name = os.path.join(saved_path, "attack_meta_data.yaml")
    save_yaml(attack_meta_data, save_name)

    return attack_meta_data

def softmax(x):
    if isinstance(x, list):
        x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
