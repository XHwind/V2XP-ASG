# -*- coding: utf-8 -*-
"""
Scenario manager for collect carla rule-based traffic manager data
"""
import random
import math
from random import shuffle
from datetime import datetime

import carla
import numpy as np

from asg.core.scene_manager.base_manager import BaseManager
from asg.core.sensor.lidar_sensor import LidarSensor
from asg.core.scenario_manager.utils import car_blueprint_filter, \
    multi_class_car_blueprint_filter
from asg.core.sensor.sensor_manager import SensorManager


class CollectionManager(BaseManager):
    def __init__(self, params):
        super().__init__(params)
        # load current time for data dumping and evaluation
        current_time = datetime.now()
        self.current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        self.load_opencda_format = params['load_opencda_format']

        # create sensor manager
        self.sensor_manager = SensorManager(self.current_time,
                                            create_folder=True)

        self.traffic_params = self.scene_params['traffic_params']
        # the center position is equal to infra
        self.center = self.traffic_params['center']
        # lidar configurations
        self.lidar_params = self.scene_params['lidar_params']
        self.rsu_lidar_params = self.scene_params['rsu_lidar_params']

        self.multi_class_bp = params['multi_class_bp']

        self.detection_class_sample_prob = params[
            'detection_class_sample_prob']

        # normalize probability
        self.detection_class_sample_prob = {
            k: v / sum(self.detection_class_sample_prob.values()) for k, v in
            self.detection_class_sample_prob.items()}

        # traffic manager
        tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            self.traffic_params['global_distance'])
        tm.set_synchronous_mode(self.traffic_params['sync_mode'])
        tm.set_osm_mode(self.traffic_params['set_osm_mode'])
        tm.global_percentage_speed_difference(
            self.traffic_params['global_speed_perc'])

        # spawn vehicles with sensor
        self.spawn_vehicle_by_range(tm)
        self.spawn_rsus()
        # dump the overall configuation

        config_name = "collection.yaml" \
            if not self.load_opencda_format else "data_protocol.yaml"
        self.sensor_manager.dump_config(self.scene_params, config_name=config_name)

    def tick(self):
        self.world.tick()
        if self.load_opencda_format:
            self.sensor_manager.dump_data_opencda()
        else:
            self.sensor_manager.dump_data()

    def spawn_rsus(self):
        for rsu_config in self.traffic_params['rsu_list']:
            coordinates = rsu_config['spawn_position']
            id = rsu_config['id']
            lidar_sensor = LidarSensor(None,
                                       self.world,
                                       self.rsu_lidar_params,
                                       global_position=coordinates)
            self.sensor_manager.add_rsu_sensor_pair(id, lidar_sensor)

    def spawn_vehicle_by_range(self, tm):
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.
        """
        blueprint_library = self.world.get_blueprint_library()
        if not self.multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library)
        else:
            label_list = list(self.detection_class_sample_prob.keys())
            prob = [self.detection_class_sample_prob[itm] for itm in
                    label_list]

        # spawn radius around the center
        radius = self.traffic_params['radius']
        # number of vehicles for spawning
        spawn_num = self.traffic_params['spawn_num']
        # stride
        step = self.traffic_params['step']
        # calculate the spawn range coordinate
        x_min, x_max, y_min, y_max = \
            math.floor(self.center[0] - radius), \
            math.ceil(self.center[0] + radius), \
            math.floor(self.center[1] - radius), \
            math.ceil(self.center[1] + radius)

        spawn_range = [x_min, x_max, y_min, y_max, step, step, spawn_num]
        spawn_set = set()

        for x in range(x_min, x_max, int(spawn_range[4])):
            for y in range(y_min, y_max, int(spawn_range[5])):
                location = carla.Location(x=x, y=y, z=0.3)
                way_point = self.carla_map.get_waypoint(location).transform

                spawn_set.add((way_point.location.x,
                               way_point.location.y,
                               way_point.location.z,
                               way_point.rotation.roll,
                               way_point.rotation.yaw,
                               way_point.rotation.pitch))
        count = 0
        spawn_list = list(spawn_set)
        shuffle(spawn_list)

        while count < spawn_num:
            if len(spawn_list) == 0:
                break

            coordinates = spawn_list[0]
            spawn_list.pop(0)
            spawn_transform = carla.Transform(carla.Location(x=coordinates[0],
                                                             y=coordinates[1],
                                                             z=coordinates[2]),
                                              carla.Rotation(
                                                  roll=coordinates[3],
                                                  yaw=coordinates[4],
                                                  pitch=coordinates[5]))
            # sample a bp from various classes
            if self.multi_class_bp:
                label = np.random.choice(label_list, p=prob)
                ego_vehicle_random_list = multi_class_car_blueprint_filter(
                    label, blueprint_library, self.bp_meta)

            ego_vehicle_bp = random.choice(ego_vehicle_random_list)
            color = None
            if ego_vehicle_bp.has_attribute("color"):
                color = random.choice(
                    ego_vehicle_bp.get_attribute('color').recommended_values)
                ego_vehicle_bp.set_attribute('color', color)
            # avid hitting the ground
            spawn_count = 0
            max_spawn_count = 20
            vehicle = self.world.try_spawn_actor(ego_vehicle_bp,
                                                 spawn_transform)

            while not vehicle:
                spawn_count += 1
                spawn_transform.location.z += 0.01
                vehicle = self.world.try_spawn_actor(ego_vehicle_bp,
                                                     spawn_transform)
                if spawn_count >= max_spawn_count:
                    break

            if not vehicle:
                continue

            # augment the original carla.vehicle with bp id and color
            vehicle.bp_id = ego_vehicle_bp.id
            vehicle.color = color if color else None

            lidar_sensor = LidarSensor(vehicle,
                                       self.world,
                                       self.lidar_params,
                                       global_position=None)
            self.sensor_manager.add_veh_sensor_pair(vehicle, lidar_sensor)

            vehicle.set_autopilot(True, 8000)
            tm.auto_lane_change(vehicle,
                                self.traffic_params['auto_lane_change'])

            if 'ignore_lights_percentage' in self.traffic_params:
                tm.ignore_lights_percentage(vehicle,
                                            self.traffic_params[
                                                'ignore_lights_percentage'])

            # each vehicle have slight different speed
            tm.vehicle_percentage_speed_difference(
                vehicle,
                self.traffic_params['global_speed_perc'] + random.randint(-10,
                                                                          10))
            count += 1

    def destroy(self):
        """
        Destroy all actors in the world.
        """
        self.sensor_manager.destroy()

    def close(self):
        """
        Simulation close.
        """
        # restore to origin setting
        self.world.apply_settings(self.origin_settings)
