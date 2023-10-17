import random
import numpy as np
import json
from datetime import datetime
from collections import OrderedDict

import carla

from asg.core.scene_manager.base_manager import BaseManager
from asg.core.sensor.lidar_sensor import LidarSensor
from asg.core.scenario_manager.utils import car_blueprint_filter
from asg.core.sensor.sensor_manager import SensorManager
from asg.utils.pcd_utils import o3d_to_np
from asg.utils.geometry_utils import get_bbx_in_world, get_2d_distance


class SceneManager(BaseManager):
    def __init__(self, params):
        super(SceneManager, self).__init__(params)

        self.num_cavs = params["v2x_setting"]["num_cavs"]
        self.num_infra = params["v2x_setting"]["num_infra"]
        self.vehicle_lidar_params = params["vehicle_lidar_params"]
        self.rsu_lidar_params = params["rsu_lidar_params"]
        self.spawn_range = params["world"]["spawn_range"]

        with open(params["bp_meta_path"]) as f:
            self.bp_meta = json.load(f)

        assert self.num_infra == 1, \
            f"Right now only 1 connected infra is allowed but received {self.num_infra}"

    def set_agent_transform(self, vid, new_transform):
        self.get_agent(vid).set_transform(new_transform)

    def get_agent(self, i):
        return self.agent_pair_id_map[i][0]

    def get_vehicle_id_list(self):
        return [itm for itm in self.agent_pair_id_map.keys() if itm >= 0]

    def get_vehicle_id_list_with_type_within_spawn_range(self, veh_type):
        veh_id_list_within_spawn_range = \
            self.get_vehicle_id_list_within_spawn_range()
        return [itm for itm in veh_id_list_within_spawn_range if self.bp_meta[
            self.agent_pair_id_map[itm][0].bp_id]['class'] == veh_type]

    def get_vehicle_id_list_within_spawn_range(self):
        veh_id_list_within_spawn_range = []
        for veh_id in self.get_vehicle_id_list():
            veh, _ = self.agent_pair_id_map[veh_id]
            veh_pose = veh.get_transform()
            veh_pose_list = [veh_pose.location.x, veh_pose.location.y,
                             veh_pose.location.z]
            # Ignore vehicles with pose outside of the spawn range of RSU
            if get_2d_distance(veh_pose_list,
                               self.rsu_center_pose) >= self.spawn_range:
                continue
            veh_id_list_within_spawn_range.append(veh_id)

        return veh_id_list_within_spawn_range

    def get_original_cav_id_list(self):
        return [k for k, v in self.agent_pair_id_map.items() if
                k >= 0 and v[0].type == "CAV"]

    def get_extended_bbx_in_world(self, vid, transform=None,
                                  dx=0.2, dy=0.2, dz=0.2):
        veh = self.get_agent(vid)
        veh_transform = veh.get_transform() if transform is None else transform
        veh_bbx = veh.bounding_box

        bbx_world = get_bbx_in_world(veh_transform, veh_bbx, dx, dy, dz)
        return bbx_world

    def initial_scene(self, scene_meta, agent_data):
        current_time = datetime.now()
        self.current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.current_sensor_manager = SensorManager(self.current_time,
                                                    path=self.save_path)

        self.current_scene_meta = scene_meta
        # overwrite the recorded lidar params with sg yaml's settings
        self.current_scene_meta["lidar_params"] = self.vehicle_lidar_params
        self.current_scene_meta["rsu_lidar_params"] = self.rsu_lidar_params

        town_name = scene_meta["world"]["town_name"]
        self.load_map(town_name)
        self.spawn_rsus_by_poses(agent_data, self.rsu_lidar_params)
        self.spawn_vehicles_by_poses(agent_data, self.vehicle_lidar_params)
        self.agent_pair_id_map = OrderedDict()

        # vehicle id starts from 0 and is positive
        for i, (veh, lidar) in enumerate(
                self.current_sensor_manager.get_veh_sensor_pair()):
            self.agent_pair_id_map[i] = (veh, lidar)

        # rsu's id starts from -1 and is negative
        # Notice that this id is different from original rid that is assigned
        # when recording
        for i, (rid, lidar) in enumerate(
                self.current_sensor_manager.get_rsu_sensor_pair()):
            self.agent_pair_id_map[-(i + 1)] = (rid, lidar)

    def initial_original_poses(self):
        self.agent_original_poses = OrderedDict()

        # vehicle id starts from 0 and is positive
        for i, (veh, lidar) in enumerate(
                self.current_sensor_manager.get_veh_sensor_pair()):
            self.agent_original_poses[i] = veh.get_transform()

    def get_observation(self, cav_id_list, rsu_id_list):
        """
        Get observations from carla server.
        Args:
            cav_id_list: list
                List of cav ids
            rsu_id_list: list
                List of rsu ids

        Returns:
            Observation: dict
                {id: o3d.geometry.PointCloud}
        """
        observation = OrderedDict()
        assert len(rsu_id_list) == 1
        for id in sorted(rsu_id_list + cav_id_list):
            lidar = self.agent_pair_id_map[id][1]
            lidar_pcd = self.current_sensor_manager.retrieve_lidar_points(
                lidar)

            lidar_pose = lidar.sensor.get_transform()
            lidar_pose = [lidar_pose.location.x,
                          lidar_pose.location.y,
                          lidar_pose.location.z,
                          lidar_pose.rotation.roll,
                          lidar_pose.rotation.yaw,
                          lidar_pose.rotation.pitch]
            ego_flag = id == rsu_id_list[0]
            observation[id] = {
                "lidar_pcd": lidar_pcd,
                "lidar_np": o3d_to_np(lidar_pcd),
                "lidar_pose": lidar_pose,
                "ego": ego_flag
            }

        return observation

    def get_gt_bbx(self):
        """
        Get ground truth annotations for vehicles within the range.
        Returns:
            gt: dict
                Dictionary of agent id and associated annotations.
        """
        gt = OrderedDict()
        for id, (agent, lidar) in self.agent_pair_id_map.items():
            # if id is negative, then it is infra and we skip it.
            if id < 0:
                continue
            assert not isinstance(agent, str) and not isinstance(agent, int)
            veh_pose = agent.get_transform()
            veh_bbx = agent.bounding_box
            # Only treat vehicle with pose within the spawn range of RSU as the
            # gt vehicle
            if get_2d_distance([veh_pose.location.x, veh_pose.location.y,
                                veh_pose.location.z],
                               self.rsu_center_pose) >= self.spawn_range:
                continue

            gt[id] = {'location': [veh_pose.location.x,
                                   veh_pose.location.y,
                                   veh_pose.location.z],
                      'center': [veh_bbx.location.x,
                                 veh_bbx.location.y,
                                 veh_bbx.location.z],
                      'extent': [veh_bbx.extent.x,
                                 veh_bbx.extent.y,
                                 veh_bbx.extent.z],
                      'angle': [veh_pose.rotation.roll,
                                veh_pose.rotation.yaw,
                                veh_pose.rotation.pitch],
                      'color': agent.color,
                      'bp_id': agent.bp_id}
        return gt

    def spawn_rsus_by_poses(self, agents, lidar_params):
        rsu_coordinate_list = []
        for rsu_id in agents.keys():
            if int(rsu_id) >= 0:
                continue
            coordinates = agents[rsu_id]["lidar_pose"]
            lidar_sensor = LidarSensor(None, self.world, lidar_params,
                                       global_position=coordinates)
            self.current_sensor_manager.add_rsu_sensor_pair(int(rsu_id),
                                                            lidar_sensor)
            rsu_coordinate_list.append(coordinates)
        assert len(rsu_coordinate_list), "Currently only 1 rsu is supported."
        self.rsu_center_pose = rsu_coordinate_list[0][:2]

    def spawn_vehicles_by_poses(self, agents, lidar_params, random_color=False,
                                random_bp=False):
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp_list = car_blueprint_filter(blueprint_library)

        # default vehicle blueprint and default vehicle color
        default_model = 'vehicle.lincoln.mkz_2017'
        default_color = '0, 255, 0'

        vehicle_bp = blueprint_library.find(default_model)

        for cav_id in agents.keys():
            if int(cav_id) < 0:
                continue
            location = agents[cav_id]["location"]
            angle = agents[cav_id]["angle"]
            pose = carla.Transform(
                carla.Location(x=location[0], y=location[1],
                               z=location[2]),
                carla.Rotation(roll=angle[0], yaw=angle[1], pitch=angle[2]))

            if "bp_id" in agents[cav_id]:
                vehicle_bp = blueprint_library.find(agents[cav_id]["bp_id"])
            elif random_bp:
                vehicle_bp = random.choice(vehicle_bp_list)

            if "color" in agents[cav_id]:
                color = agents[cav_id]["color"]
            else:
                color = default_color

            if vehicle_bp.has_attribute("color"):
                if random_color and color is None:
                    color = random.choice(
                        vehicle_bp.get_attribute("color").recommended_values)
                vehicle_bp.set_attribute("color", color)

            vehicle = self.world.try_spawn_actor(vehicle_bp, pose)
            # avid hitting the ground
            spawn_count = 0
            max_spawn_count = 1000
            while not vehicle:
                spawn_count += 1
                pose.location.z += 0.01
                vehicle = self.world.try_spawn_actor(vehicle_bp, pose)
                if spawn_count >= max_spawn_count:
                    break

            if not vehicle:
                # raise RuntimeError(
                #     f"Fail to spawn vehicle: {cav_id} with pose: {pose}")
                print(f"cav_id: {cav_id}: {agents[cav_id]}")
                print(
                    f"Fail to spawn vehicle{cav_id}: pose -- {pose}, type -- {agents['Type']}")
                continue
            vehicle.bp_id = vehicle_bp.id
            vehicle.color = color
            vehicle.ego = agents[cav_id]["ego"]
            if "Type" in agents[cav_id]:
                vehicle.type = agents[cav_id]["Type"]

            lidar_sensor = LidarSensor(vehicle, self.world, lidar_params,
                                       global_position=None)
            self.current_sensor_manager.add_veh_sensor_pair(vehicle,
                                                            lidar_sensor)

    def clean(self):
        self.current_sensor_manager.destroy()
        self.world.apply_settings(self.origin_settings)
        self.current_scene_meta = None
        self.current_sensor_manager = None
        self.current_time = None
        self.agent_pair_id_map = None
        self.agent_original_poses = None
        # clean saving path related names
        self.scenario_name = None
        self.timstamp = None
        self.world.tick()

    def tick(self, save_current_frame=False):
        self.world.tick()
        if save_current_frame:
            self.current_sensor_manager.dump_data()
            self.current_sensor_manager.dump_config(self.current_scene_meta)

    def set_scenario_name_and_timestamp(self, scenario_name, timestamp):
        self.scenario_name = scenario_name
        self.timestamp = timestamp

    def get_scenario_name_and_timstamp(self):
        return self.scenario_name, self.timestamp
