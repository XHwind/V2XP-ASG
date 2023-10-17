import sys
import random
import json

import carla
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BaseManager:
    """
    Base class for managing SceneManager and CollectionManager

    Args:
        scenario_params: dict
            Dictionary for parameters

    Attributes:
        fixed_delta_seconds: int
            The simulation time interval between two simulation time.
        weather_config: dict
            Dictionary for the weather configuration with default None (use carla default weather).
        client: carla.Client
            Carla client object.
        town_name: str
            The map name to be loaded.
        world: carla.World
            Carla world object.
        carla_map: carla.Map
            Carla map object.
        save_path: str
            The save path.
    """

    def __init__(self, scenario_params):
        # this defines carla world sync mode, weather, town name, and seed.
        self.scene_params = scenario_params
        simulation_config = scenario_params["world"]
        # set random seed if stated
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])
        self.fixed_delta_seconds = simulation_config['fixed_delta_seconds']
        # assume global weather setting for all scenes for now
        self.weather_config = simulation_config[
            'weather'] if "weather" in simulation_config else None

        # bbx/blueprint meta
        with open(scenario_params['bp_meta_path']) as f:
            self.bp_meta = json.load(f)

        # setup the carla client
        self.client = \
            carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)

        if "town_name" in simulation_config:
            self.town_name = simulation_config["town_name"]
        else:
            self.town_name = "Town10HD"

        try:
            self.load_map(self.town_name)
        except RuntimeError:
            print(
                f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                f"Please download all town maps to your CARLA "
                f"repo!{bcolors.ENDC}" % simulation_config['town_name'])
        if not self.world:
            sys.exit('World loading failed')

        self.carla_map = self.world.get_map()
        self.save_path = simulation_config[
            "save_path"] if "save_path" in simulation_config else None

    def set_spectator(self, center):
        location = carla.Location(x=center[0], y=center[1], z=80)
        rotation = carla.Rotation(pitch=-90)
        transform = carla.Transform(location, rotation)
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)

    def tick(self, save_current_frame=False):
        pass

    def load_map(self, town_name):
        """
        Load carla map and setup basic configurations.
        Args:
            town_name : map name
        """
        self.world = self.client.load_world(town_name)
        self.origin_settings = self.world.get_settings()

        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(new_settings)
        if self.weather_config != None:
            weather = self.set_weather(self.weather_config)
            self.world.set_weather(weather)



    @staticmethod
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather
