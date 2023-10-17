# -*- coding: utf-8 -*-
"""
Sensor manager for data dumping
"""

import os
import yaml
from collections import OrderedDict

import open3d as o3d
import numpy as np

from asg.core.sensor.utils import get_speed
from asg.hypes_yaml.yaml_utils import save_yaml
from asg.utils.pcd_utils import o3d_to_np
from asg.data_utils.scene_datasets.data_dumping_utils import dump_data_opencda, \
    dump_agent_choice


class SensorManager:
    def __init__(self, save_time, path=None, create_folder=False):
        self.veh_sensor_pair = []
        self.rsu_sensor_pair = []
        self.save_time = save_time
        if path is None:
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.save_parent_folder = os.path.join(current_path,
                                                   '../../../data_dumping',
                                                   save_time)
        else:
            self.save_parent_folder = os.path.join(path, save_time)

        if create_folder and not os.path.exists(self.save_parent_folder):
            os.makedirs(self.save_parent_folder)

    def add_veh_sensor_pair(self, vehicle, sensor):
        """
        Add vehicle sensor pair to list.

        Parameters
        ----------
        vehicle : carla.Actor
            The vehicle with lidar
        sensor : asg.sensor.LidarSensor
        """
        self.veh_sensor_pair.append((vehicle, sensor))

    def add_rsu_sensor_pair(self, id, sensor):
        """
        Add rsu ID and sensor pair to list.

        Parameters
        ----------
        id : int
           RSU ID
        sensor : asg.sensor.LidarSensor
        """
        self.rsu_sensor_pair.append((id, sensor))

    def get_veh_sensor_pair(self):
        """
        Get vehicle sensor pairs
        Returns:
            List of (carla.Actor, asg.sensor.LidarSensor)
        """
        return self.veh_sensor_pair

    def get_rsu_sensor_pair(self):
        """
        Get rsu sensor pairs
        Returns:
            List of (int, asg.sensor.LidarSensor)
        """
        return self.rsu_sensor_pair

    def get_vehicles(self):
        """
        Get vehicles
        Returns:
            List of carla.Actor
        """
        return [x[0] for x in self.veh_sensor_pair]

    def dump_config(self, scene_params,
                    config_name="collection.yaml",
                    save_path=None):
        """
        Dump the overall configuration yaml file.
        """
        save_path = self.save_parent_folder if save_path is None else save_path
        save_yaml(scene_params, os.path.join(save_path,
                                             config_name))

    def dump_data(self):
        """
        Dump the pcd and yaml files.
        """
        # todo dump the data in OpenCDA format
        # todo change the self.save_parent_folder to resume_attack folder
        for vehicle, lidar in self.veh_sensor_pair:
            dump_yml = {}

            frame = lidar.frame

            veh_id = vehicle.id
            veh_spd = get_speed(vehicle)
            veh_pose = vehicle.get_transform()
            veh_bbx = vehicle.bounding_box

            lidar_pose = lidar.sensor.get_transform()
            lidar_pcd = self.retrieve_lidar_points(lidar)

            vehicle_folder = os.path.join(self.save_parent_folder,
                                          str(veh_id))
            if not os.path.exists(vehicle_folder):
                os.makedirs(vehicle_folder)

            dump_yml.update({'id': veh_id,
                             'bp_id': vehicle.bp_id,
                             'color': vehicle.color,
                             'speed': veh_spd,
                             'location': [veh_pose.location.x,
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
                             'lidar_pose': [
                                 lidar_pose.location.x,
                                 lidar_pose.location.y,
                                 lidar_pose.location.z,
                                 lidar_pose.rotation.roll,
                                 lidar_pose.rotation.yaw,
                                 lidar_pose.rotation.pitch]})
            # dump pcd file
            pcd_name = '%06d' % frame + '.pcd'
            o3d.io.write_point_cloud(os.path.join(vehicle_folder,
                                                  pcd_name),
                                     pointcloud=lidar_pcd,
                                     write_ascii=True)
            # dump meta data
            meta_name = os.path.join(vehicle_folder,
                                     '%06d' % frame + '.yaml')

            with open(meta_name, 'w') as outfile:
                yaml.dump(dump_yml, outfile, default_flow_style=False)
        for rid, lidar in self.rsu_sensor_pair:
            dump_yml = {}
            frame = lidar.frame
            lidar_pose = lidar.sensor.get_transform()
            lidar_pcd = self.retrieve_lidar_points(lidar)

            rsu_folder = os.path.join(self.save_parent_folder,
                                      str(rid))
            if not os.path.exists(rsu_folder):
                os.makedirs(rsu_folder)

            dump_yml.update({'id': rid,
                             'lidar_pose': [
                                 lidar_pose.location.x,
                                 lidar_pose.location.y,
                                 lidar_pose.location.z,
                                 lidar_pose.rotation.roll,
                                 lidar_pose.rotation.yaw,
                                 lidar_pose.rotation.pitch]})
            # dump pcd file
            pcd_name = '%06d' % frame + '.pcd'
            o3d.io.write_point_cloud(os.path.join(rsu_folder,
                                                  pcd_name),
                                     pointcloud=lidar_pcd,
                                     write_ascii=True)
            # dump meta data
            meta_name = os.path.join(rsu_folder,
                                     '%06d' % frame + '.yaml')
            with open(meta_name, 'w') as outfile:
                yaml.dump(dump_yml, outfile, default_flow_style=False)

    def dump_data_opencda(self):
        """
        Dump the pcd and yaml files in opencda format.
        """
        gt = OrderedDict()
        for vehicle, lidar in self.veh_sensor_pair:
            veh_pose = vehicle.get_transform()
            veh_bbx = vehicle.bounding_box
            veh_id = int(vehicle.id)

            gt[veh_id] = {'location': [veh_pose.location.x,
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
                          'true_ego_pos': [
                              veh_pose.location.x,
                              veh_pose.location.y,
                              veh_pose.location.z,
                              veh_pose.rotation.roll,
                              veh_pose.rotation.yaw,
                              veh_pose.rotation.pitch],
                          'bp_id': vehicle.bp_id,
                          'color': vehicle.color,
                          }
        observations = OrderedDict()
        for (agent, lidar) in self.rsu_sensor_pair + self.veh_sensor_pair:
            lidar_pose = lidar.sensor.get_transform()
            lidar_pose = [lidar_pose.location.x,
                          lidar_pose.location.y,
                          lidar_pose.location.z,
                          lidar_pose.rotation.roll,
                          lidar_pose.rotation.yaw,
                          lidar_pose.rotation.pitch]
            lidar_pcd = self.retrieve_lidar_points(lidar)
            frame = lidar.frame
            if isinstance(agent, int) or isinstance(agent, str):
                agent_id = int(agent)
            else:
                agent_id = int(agent.id)
            observations[agent_id] = {
                "lidar_pcd": lidar_pcd,
                "lidar_np": o3d_to_np(lidar_pcd),
                "lidar_pose": lidar_pose,
            }
        save_folder = "/".join(self.save_parent_folder.split("/")[:-1])
        scenario_name = self.save_parent_folder.split("/")[-1]
        timestamp = '%06d' % frame
        agent_list = list(observations.keys())
        dump_data_opencda(save_folder, scenario_name, timestamp, agent_list,
                          observations, gt, sub_folder="")

    @staticmethod
    def retrieve_lidar_points(lidar):
        """
        Save 3D lidar points to disk.
        """
        point_cloud = lidar.data
        while point_cloud is None:
            point_cloud = lidar.data
            continue

        frame = lidar.frame

        point_xyz = point_cloud[:, :-1]
        point_intensity = point_cloud[:, -1]
        point_intensity = np.c_[
            point_intensity,
            np.zeros_like(point_intensity),
            np.zeros_like(point_intensity)
        ]

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

        return o3d_pcd

    def destroy(self):
        for vehicle, lidar in self.veh_sensor_pair:
            vehicle.destroy()
            lidar.sensor.destroy()

        for id, lidar in self.rsu_sensor_pair:
            lidar.sensor.destroy()
