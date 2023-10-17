import os
from collections import OrderedDict

from asg.core.scene_manager.scene_manager import SceneManager
from asg.data_utils.scene_datasets.base_scene_dataset import BaseSceneDataset
from asg.hypes_yaml.yaml_utils import load_yaml
from asg.utils.json_utils import load_filtered_scene_list


class SceneDataset(BaseSceneDataset):
    def __init__(self, params):
        super(SceneDataset, self).__init__(params)
        # The pairs contain the (scenario name, timestamp) pairs which can be
        # used to define a sequence of scenes.
        # The below line is used to load preprocessed sub-sampled scenes
        # self.scenario_time_pairs = load_filtered_scene_list(params["filtered_scene_path"])
        self.scenario_time_pairs = self.load_all_scenario_time_pairs(params["data_path"])
        # The index count the current scene's position in the
        # scenario time pairs.
        self.index = 0
        self.scene_manager = SceneManager(params)

    def __len__(self):
        return len(self.scenario_time_pairs)

    def __iter__(self):
        return self

    def __getitem__(self, i):
        scenario_name, time = self.scenario_time_pairs[i]
        print(scenario_name, time)
        # From scenario database (scenario_name, agent_id, timestamp) to get
        # scene database (scenario_name, agent_id) with the given the time
        scene_database = self.get_scene_database(scenario_name, time)
        scene_meta = load_yaml(scene_database["collection_meta"])
        if self.opencda_format:
            data = self.load_opencda_format_data_dict(scene_database)
            scene_meta["traffic_params"] = {}
            scene_meta["traffic_params"]["center"] = data["-1"]["lidar_pose"][
                                                     :2] + [0]
        else:
            data = OrderedDict()
            for cav_id in scene_database["agents"].keys():
                data[cav_id] = OrderedDict()
                data[cav_id]["ego"] = scene_database["agents"][cav_id]["ego"]
                agent_meta = load_yaml(
                    scene_database["agents"][cav_id]["yaml"])
                # For RSU the location and angle are stored in lidar_pose
                if int(cav_id) >= 0:
                    data[cav_id]["location"] = agent_meta["location"]
                    data[cav_id]["angle"] = agent_meta["angle"]
                    if "bp_id" in agent_meta:
                        data[cav_id]["bp_id"] = agent_meta["bp_id"]
                    if "color" in agent_meta:
                        data[cav_id]["color"] = agent_meta["color"]
                data[cav_id]["lidar_pose"] = agent_meta["lidar_pose"]

        # Load scene's HD map and spawn agents (i.e., vehicles and infra).
        self.scene_manager.initial_scene(scene_meta, data)
        self.scene_manager.set_scenario_name_and_timestamp(scenario_name, time)
        self.scene_manager.set_spectator(
            scene_meta["traffic_params"]["center"])
        for i in range(10):
            self.scene_manager.tick()

        self.scene_manager.initial_original_poses()

        return self.scene_manager

    def clean(self):
        self.scene_manager.clean()

    def __next__(self):
        self.index = self.index if self.index >= 0 else self.index + len(self)
        if self.index < len(self):
            self.index += 1
            return self[self.index - 1]

        raise StopIteration

    def load_opencda_format_data_dict(self, scene_database):
        data = OrderedDict()
        rsu_cav_id_list = list(scene_database["agents"].keys())
        # load CAVs information
        for cav_id in scene_database["agents"].keys():
            data[cav_id] = OrderedDict()
            # ego flag
            data[cav_id]["ego"] = scene_database["agents"][cav_id]["ego"]
            agent_meta = load_yaml(scene_database["agents"][cav_id]["yaml"])
            if int(cav_id) >= 0:
                data[cav_id]["location"] = agent_meta["true_ego_pos"][:3]
                data[cav_id]["angle"] = agent_meta["true_ego_pos"][3:]
                if "bp_id" in agent_meta:
                    data[cav_id]["bp_id"] = agent_meta["bp_id"]
                if "color" in agent_meta:
                    data[cav_id]["color"] = agent_meta["color"]
            data[cav_id]["lidar_pose"] = agent_meta["lidar_pose"]
            data[cav_id]["Type"] = "CAV" if int(cav_id) >= 0 else "RSU"
            # load background vehicles' information
            for vid in agent_meta["vehicles"]:
                # If vid is ego agent, skip it
                if str(vid) in scene_database["agents"].keys():
                    if str(vid) in data and \
                            "bp_id" in agent_meta["vehicles"][vid]:
                        data[str(vid)]["bp_id"] = agent_meta["vehicles"][vid][
                            "bp_id"]
                    if str(vid) in data and \
                            "color" in agent_meta["vehicles"][vid]:
                        data[str(vid)]["color"] = \
                            agent_meta["vehicles"][vid]["color"]
                    continue
                data[str(vid)] = OrderedDict()
                data[str(vid)]["ego"] = False
                data[str(vid)]["location"] = agent_meta["vehicles"][vid][
                    "location"]
                data[str(vid)]["angle"] = agent_meta["vehicles"][vid]["angle"]
                data[str(vid)]["Type"] = "BackGroundVehicle"
                if "bp_id" in agent_meta["vehicles"][vid]:
                    data[str(vid)]["bp_id"] = agent_meta["vehicles"][vid][
                        "bp_id"]
                if "color" in agent_meta["vehicles"][vid]:
                    data[str(vid)]["color"] = agent_meta["vehicles"][vid][
                        "color"]
        return data

    def load_all_scenario_time_pairs(self, data_folder):
        scenario_time_pairs = []
        for scenario_name in sorted(os.listdir(data_folder)):
            scenario_path = os.path.join(data_folder, scenario_name)
            overlap_timestamp_list = []
            for cav_id_name in sorted(os.listdir(scenario_path)):
                agent_path = os.path.join(scenario_path, cav_id_name)
                if not os.path.isdir(agent_path):
                    continue
                agent_timestamp_list = [x.split(".")[0] for x in
                                        os.listdir(agent_path)]
                agent_timestamp_list = sorted(set(agent_timestamp_list))
                if len(overlap_timestamp_list) == 0:
                    overlap_timestamp_list = agent_timestamp_list
                else:
                    overlap_timestamp_list = list(sorted(set(agent_timestamp_list +
                                                     overlap_timestamp_list)))
            for timestamp in sorted(overlap_timestamp_list):
                scenario_time_pairs.append((scenario_name, timestamp))
        return scenario_time_pairs
