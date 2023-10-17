import math

def get_speed(vehicle, meters=False):
    """
    Compute speed of a vehicle in Km/h.

    Parameters
    ----------
    meters : bool
        Whether to use m/s (True) or km/h (False).

    vehicle : carla.vehicle
        The vehicle for which speed is calculated.

    Returns
    -------
    speed : float
        The vehicle speed.
    """
    vel = vehicle.get_velocity()
    vel_meter_per_second = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    return vel_meter_per_second if meters else 3.6 * vel_meter_per_second
