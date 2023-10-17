"""
Used to dump data.
"""
import os
from collections import OrderedDict

import open3d as o3d
import pickle

from opencood.hypes_yaml.yaml_utils import save_yaml


def dump_data_opencda(save_folder, scenario_name, timestamp, agent_list,
                      observations, gt, sub_folder="generated_scenes"):
    """
    Dump the data in to te folder with the same format as OpenCDA.

    Parameters
    ----------
    save_folder : str
        The output root folder.
    scenario_name : str
        The scenario name ( folder name) for storing the scenes
    timestamp : str
        The timestamp associated with the scene
    agent_list : list
        The ids of cav and rsu.
    observations : dict
        The lidar observations.
    gt : dict
        The dictionary that contains all dynamic objects.
    """
    if len(sub_folder):
        save_folder = os.path.join(save_folder, sub_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    scene_folder = os.path.join(save_folder, scenario_name)

    for id_ in agent_list:
        agent_folder = os.path.join(scene_folder, str(id_))

        if not os.path.exists(agent_folder):
            os.makedirs(agent_folder)

        # retrieve the corresponding lidar info
        lidar_pcd = observations[id_]['lidar_pcd']
        lidar_pose = observations[id_]['lidar_pose']

        # gt information
        gt_dict = OrderedDict({'vehicles': {},
                               'lidar_pose': lidar_pose})
        if id_ >= 0:
            gt_dict['true_ego_pos'] = gt[id_]['location'] + gt[id_]["angle"]
            gt_dict['bp_id'] = gt[id_]['bp_id']
            gt_dict['color'] = gt[id_]['color']


        for veh_id, veh in gt.items():
            if veh_id == id_:
                continue
            gt_dict['vehicles'].update({veh_id: veh})

        # write to pcd file
        pcd_name = timestamp + '.pcd'
        o3d.io.write_point_cloud(os.path.join(agent_folder,
                                              pcd_name),
                                 pointcloud=lidar_pcd,
                                 write_ascii=True)

        yaml_name = timestamp + '.yaml'
        save_yaml(gt_dict, os.path.join(agent_folder, yaml_name))


def dump_agent_choice(save_folder, scenario_name, timestamp, rsu_id_list,
                      cav_id_list):
    save_folder = os.path.join(save_folder, "agent_choice")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder,
                             f"{scenario_name}_{timestamp}.pickle")
    data = {
        "rsu_id_list": rsu_id_list,
        "cav_id_list": cav_id_list
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def dump_agent_weights(save_folder, scenario_name, timestamp, weight_dict):
    save_folder = os.path.join(save_folder, "agent_weight_dir")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder,
                             f"{scenario_name}_{timestamp}.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(weight_dict, f)



def load_agent_list(save_folder, scenario_name, timestamp):
    path = os.path.join(save_folder, f"{scenario_name}_{timestamp}.pickle")
    with open(path, "rb") as f:
        data = pickle.load(f)
        rsu_id_list = data["rsu_id_list"]
        cav_id_list = data["cav_id_list"]

    rsu_id_list = sorted(list(rsu_id_list))
    cav_id_list = sorted(list(cav_id_list))
    return rsu_id_list, cav_id_list

def load_agent_weight(save_folder, scenario_name, timestamp):
    path = os.path.join(save_folder, f"{scenario_name}_{timestamp}.pickle")
    with open(path, "rb") as f:
        weight_dict = pickle.load(f)
    return weight_dict
