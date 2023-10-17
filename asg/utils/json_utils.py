import json


def load_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, save_name):
    """
    Save the dictionary into a json file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output json file.
    """
    with open(save_name, "w") as f:
        f.write(json.dumps(data, indent=4, separators=(',', ': ')))


def load_filtered_scene_list(path):
    """
    Load filtered scene list from the given path
    Args:
        path: str
            path to the json file.

    Returns:
        filtered_scene_list: list
            List of (scenario_name, time) pairs.

    """
    with open(path, "r") as f:
        data = json.load(f)

    filtered_scene_list = []
    for scenario_name in sorted(data.keys()):
        for time in sorted(data[scenario_name]):
            filtered_scene_list.append((scenario_name, time))

    return filtered_scene_list
