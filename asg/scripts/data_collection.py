# -*- coding: utf-8 -*-
"""
Data collection for rule-based traffic
"""

import carla

from asg.core.scenario_manager.collection_manager import CollectionManager
from asg.hypes_yaml.yaml_utils import load_yaml


def run_scenario(opt):
    try:
        scene_params = load_yaml(opt.config_yaml)
        scene_params['load_opencda_format'] = opt.load_opencda_format
        # create scenario manager
        scenario_manager = CollectionManager(scene_params)

        spectator = scenario_manager.world.get_spectator()

        count = 0
        max_iter = 70
        while count < max_iter:
            print(count)
            center = scene_params['traffic_params']['center']
            transform = carla.Transform(carla.Location(x=center[0],
                                                       y=center[1],
                                                       z=80),
                                        carla.Rotation(pitch=-90))
            spectator.set_transform(transform)
            scenario_manager.tick()
            count += 1

    finally:
        scenario_manager.destroy()
        scenario_manager.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="SG data initial collection")
    parser.add_argument('--config_yaml', type=str,
                        default='../hypes_yaml/collection.yaml',
                        help='data collection hyperparameter file path')
    parser.add_argument('--load_opencda_format', action='store_true',
                        help='Set if loading OpenCDA format data')
    opt = parser.parse_args()
    try:
        run_scenario(opt)
    except KeyboardInterrupt:
        print(' - Exited by user.')
