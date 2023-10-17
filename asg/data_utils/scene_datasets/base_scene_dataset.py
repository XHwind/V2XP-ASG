import os

from collections import OrderedDict


class BaseSceneDataset:
    """
    Base class for loading meta information about scenes from the disk.
    Args:
        params: dict
            Configuration parameters.

    Attributes:
        scenario_database: OrderedDict
            Store the path to files specified by scenario_name, id, timestamp etc.

    """

    def __init__(self, params):
        root_dir = params["data_path"]
        self.opencda_format = params["opencda_format"]
        # first load all paths of different scenarios
        scenario_folders = sorted([x for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_database = OrderedDict()
        self.len_record = []
        # loop over all scenarios
        for scenario_name in scenario_folders:
            scenario_folder = os.path.join(root_dir, scenario_name)
            self.scenario_database.update({scenario_name: OrderedDict()})
            collection_meta_file_name = "collection.yaml" \
                if not self.opencda_format else "data_protocol.yaml"
            self.scenario_database[scenario_name][
                "collection_meta"] = os.path.join(scenario_folder,
                                                  collection_meta_file_name)
            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            while int(cav_list[0]) < 0 and len(cav_list):
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):

                self.scenario_database[scenario_name][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml')])
                timestamps = [x.split("/")[-1].replace(".yaml", "") for x in
                              yaml_files]

                for timestamp in timestamps:
                    self.scenario_database[scenario_name][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')

                    self.scenario_database[scenario_name][cav_id][timestamp][
                        'yaml'] = \
                        yaml_file
                    self.scenario_database[scenario_name][cav_id][timestamp][
                        'lidar'] = \
                        lidar_file
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    self.scenario_database[scenario_name][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[scenario_name][cav_id][
                        'ego'] = False

    def get_scene_database(self, scenario_name, timestamp):
        scene_database = OrderedDict()
        scene_database["agents"] = OrderedDict()
        scene_database["collection_meta"] = \
            self.scenario_database[scenario_name]["collection_meta"]
        for cav_id in self.scenario_database[scenario_name].keys():
            if cav_id == "collection_meta":
                continue
            if timestamp not in self.scenario_database[scenario_name][cav_id]:
                continue
            scene_database["agents"][cav_id] = \
                self.scenario_database[scenario_name][cav_id][timestamp]
            scene_database["agents"][cav_id]["ego"] = \
                self.scenario_database[scenario_name][cav_id]['ego']

        return scene_database
