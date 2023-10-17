import numpy as np

from opencood.utils.box_utils import create_bbx
from opencood.utils.transformation_utils import x_to_world, x1_to_x2

OCC_DIST = 0.1


def get_2d_distance(pose1, pose2):
    assert isinstance(pose1, list) and isinstance(pose2, list)
    return np.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)


def get_bbx_in_world(veh_transform, veh_bbx, dx, dy, dz):
    veh_pose = [veh_transform.location.x + veh_bbx.location.x,
                veh_transform.location.y + veh_bbx.location.y,
                veh_transform.location.z + veh_bbx.location.z,
                veh_transform.rotation.roll, veh_transform.rotation.yaw,
                veh_transform.rotation.pitch]
    extent = [veh_bbx.extent.x + dx, veh_bbx.extent.y + dy,
              veh_bbx.extent.z + dz]

    # shape (3, 8)
    bbx = create_bbx(extent).T
    veh_to_world = x_to_world(veh_pose)

    # shape (4, 8)
    bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
    bbx_world = np.dot(veh_to_world, bbx).T
    return bbx_world


def sort_agent_order_by_occlusion_level(k, veh_dict, vid_list,
                                        rsu_id_list, cav_id_list):
    assert k <= len(vid_list)
    bbx_dict = {}
    lidar_pose_dict = {}

    for vid in sorted(veh_dict.keys()):
        if vid >= 0:
            veh_transform = veh_dict[vid].get_transform()
            veh_bbx = veh_dict[vid].bounding_box
            bbx_np = get_bbx_in_world(veh_transform, veh_bbx, 0.2, 0.2, 0.2)
            bbx_dict[vid] = bbx_np
            lidar_pose = [veh_transform.location.x + veh_bbx.location.x,
                          veh_transform.location.y + veh_bbx.location.y,
                          veh_transform.location.z + veh_bbx.location.z,
                          veh_transform.rotation.roll,
                          veh_transform.rotation.yaw,
                          veh_transform.rotation.pitch]
            lidar_pose_dict[vid] = lidar_pose
        else:
            rsu_transform = veh_dict[vid].sensor.get_transform()
            lidar_pose = [rsu_transform.location.x,
                          rsu_transform.location.y,
                          rsu_transform.location.z,
                          rsu_transform.rotation.roll,
                          rsu_transform.rotation.yaw,
                          rsu_transform.rotation.pitch]
            lidar_pose_dict[vid] = lidar_pose
    adj_mask = build_adj_matrix(lidar_pose_dict, a=35, b=35)

    occ_mask = np.zeros((len(vid_list), len(veh_dict.keys())))
    occ_ed_mask = np.zeros(len(vid_list))
    # right now exclude cavs
    for i, vid_i in enumerate(vid_list):
        if vid_i in cav_id_list:
            continue
        agent_i_lidar_pose = lidar_pose_dict[vid_i]
        bbx_i = bbx_dict[vid_i]
        for j, vid_j in enumerate(sorted(veh_dict.keys())):
            if vid_j < 0 or vid_i == vid_j or (not adj_mask[(vid_i, vid_j)]) \
                    or vid_j in cav_id_list:
                continue
            agent_j_lidar_pose = lidar_pose_dict[vid_j]
            bbx_j = bbx_dict[vid_j]
            for cav_id in rsu_id_list + cav_id_list:
                cav_lidar_pose = lidar_pose_dict[cav_id]

                T_i_to_cav = x1_to_x2(agent_i_lidar_pose, cav_lidar_pose)
                T_j_to_cav = x1_to_x2(agent_j_lidar_pose, cav_lidar_pose)

                bbx_i_rel = np.dot(T_i_to_cav, bbx_i.T).T
                bbx_j_rel = np.dot(T_j_to_cav, bbx_j.T).T
                occ_stats = occlusion_query_between_bbx1_and_bbx2(bbx_i_rel,
                                                                  bbx_j_rel)
                if occ_stats == 1:
                    occ_mask[i, j] += 1
                elif occ_stats == -1:
                    occ_ed_mask[i] += 1
                else:
                    continue
    # Index-based occlusion score map
    occ_score = occ_mask.sum(-1) + (occ_ed_mask > 0)
    # Convert occlusion score map from index-based to vid-based
    occ_score_map = {}
    for i, vid in enumerate(vid_list):
        occ_score_map[vid] = occ_score[i]


    return occ_score_map


def check_if_agent1_in_ellipse_of_agent2(agent1, agent2, a, b):
    """
    Check if agent1 is within ellipse of agent2
    Args:
        agent1: numpy.ndarray
            (N, 1, 3) or (N, N, 3)
        agent2: numpy.ndarray
            (N, 1, 3) or (N, N, 3)
        a: float
            a in ellipse equation
        b: float
            b in ellipse equation

    Returns:
        mask: numpy.ndarray
            (N, N) mask_{ij} is True if agent i within the ellipse of agent j

    """
    x1, y1 = agent1[..., 0], agent1[..., 1]
    x2, y2 = agent2[..., 0], agent2[..., 1]
    mask = ((x1 - x2) ** 2 / (a ** 2) + (y1 - y2) ** 2 / (b ** 2)) < 1
    return mask


def build_adj_matrix(agent_pose_dict, a=20, b=20):
    """
    Build adjacency matrix for the agents. Two agents are considered connected,
    if they are in epplise of each other.
    Args:
        agent_pose_dict: dict
            {vid: list (lidar pose with length of 6)}
        a: float
            a in ellipse equation
        b: float
            b in ellipse equation

    Returns:
        adj: dict
            Ajacency matrix with format {(vid_i, vid_j): Boolean} True if two
            agents are adjacent

    """
    vid_list = sorted(agent_pose_dict.keys())
    poses = np.stack([agent_pose_dict[k] for k in vid_list])
    adj = {}
    mask = check_if_agent1_in_ellipse_of_agent2(np.expand_dims(poses, axis=1),
                                                np.expand_dims(poses, axis=0),
                                                a, b)
    for i, vid_i in enumerate(vid_list):
        for j, vid_j in enumerate(vid_list):
            adj[(vid_i, vid_j)] = mask[i, j]
    return adj


def occlusion_query_between_bbx1_and_bbx2(bbx1, bbx2):
    """
    Occlusion query between bbx1 and bbx2
    Args:
        bbx1: numpy.ndarray
            Bounding box coordinates with shape (B,3).
        bbx2: numpy.ndarray
            Bounding box coordinates with shape (B,3).

    Returns:
        flag: boolean
            0 if two boxes doesn't occlude each other.
            1 if bbx1 occludes bbx2
            -1 if bbx1 is occluded by bbx2

    """

    angle_range1 = get_angle_range_for_bbx(bbx1)
    angle_range2 = get_angle_range_for_bbx(bbx2)
    if check_if_angle_range_overlap(angle_range1, angle_range2):
        dist1 = np.linalg.norm(bbx1[:, :2], axis=1).mean()
        dist2 = np.linalg.norm(bbx2[:, :2], axis=1).mean()
        # make sure it is not parallel but touches
        if dist1 <= dist2 - OCC_DIST:
            return 1
        if dist2 <= dist1 - OCC_DIST:
            return -1
    return 0


def get_angle_range_for_bbx(bbx):
    """
    Get ray-casting angle range for bbx in the x-y plane.
    Args:
        bbx: numpy.ndarray
            Bounding box coordinates with shape (B,3).

    Returns:
        angle_range: tuple
            (min angle, max angle)

    """
    # get angles for arctan(y/x)
    angles = np.arctan2(bbx[:, 1], bbx[:, 0])
    # shift to range [0, 2pi]
    angles[angles < 0] += 2 * np.pi
    min_angle = angles.min()
    max_angle = angles.max()
    if (max_angle - min_angle) > np.pi:
        max_angle, min_angle = min_angle, max_angle

    return (min_angle, max_angle)


def check_if_angle_range_overlap(angle_range1, angle_range2):
    """
    Check if the angle_range1 and angle_range2 overlaps
    Args:
        angle_range1: tuple
            Tuple of (max_angle, min_angle)
        angle_range2: tuple
            Tuple of (max_angle, min_angle)

    Returns:
        True if the two ranges overlap

    """
    return angle_between(angle_range1[0], *angle_range2) or \
           angle_between(angle_range1[1], *angle_range2)


def angle_between(n, a, b):
    """
    Check if the angle n is between angle a and angle b
    Args:
        n: float
            Angle in range (0, 2pi)
        a: float
            Angle in range (0, 2pi)
        b: float
            Angle in range (0, 2pi)

    Returns:
        True if n is between angle a and angle b

    """
    if a < b:
        return a <= n and n <= b
    return a <= n or n >= b


if __name__ == "__main__":
    agents = {16: [0, 0, 0, 0, 0, 0], 10: [10, 10, 6, 0, 0, 0],
              8: [-10, -10, 0, 0, 0, 0]}
    a = 20
    b = 20
    out = build_adj_matrix(agents, a=a, b=b)
    print(out)
