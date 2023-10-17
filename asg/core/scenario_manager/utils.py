import carla


def car_blueprint_filter(blueprint_library):
    """
    Exclude the uncommon vehicles from the default CARLA blueprint library
    (i.e., isetta, carlacola, cybertruck, t2).

    Parameters
    ----------
    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    Returns
    -------
    blueprints : list
        The list of suitable blueprints for vehicles.
    """

    blueprints = [
        blueprint_library.find('vehicle.audi.a2'),
        blueprint_library.find('vehicle.audi.tt'),
        blueprint_library.find('vehicle.dodge.charger_police'),
        blueprint_library.find('vehicle.dodge.charger_police_2020'),
        blueprint_library.find('vehicle.dodge.charger_2020'),
        blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
        blueprint_library.find('vehicle.chevrolet.impala'),
        blueprint_library.find('vehicle.mini.cooper_s'),
        blueprint_library.find('vehicle.audi.etron'),
        blueprint_library.find('vehicle.mercedes.coupe'),
        blueprint_library.find('vehicle.mercedes.coupe_2020'),
        blueprint_library.find('vehicle.bmw.grandtourer'),
        blueprint_library.find('vehicle.toyota.prius'),
        blueprint_library.find('vehicle.citroen.c3'),
        blueprint_library.find('vehicle.ford.mustang'),
        blueprint_library.find('vehicle.tesla.model3'),
        blueprint_library.find('vehicle.lincoln.mkz_2017'),
        blueprint_library.find('vehicle.lincoln.mkz_2020'),
        blueprint_library.find('vehicle.seat.leon'),
        blueprint_library.find('vehicle.nissan.patrol'),
        blueprint_library.find('vehicle.nissan.micra')
    ]

    return blueprints


def multi_class_car_blueprint_filter(label, blueprint_library, bp_meta):
    blueprints = [
        blueprint_library.find(k)
        for k, v in bp_meta.items() if v["class"] == label
    ]
    return blueprints
