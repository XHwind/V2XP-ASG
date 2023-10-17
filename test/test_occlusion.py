import numpy as np
from asg.utils.geometry_utils import occlusion_query_between_bbx1_and_bbx2

occlusion_message = {
    0: "No occlusion",
    1: "bbx1 occludes bbx2",
    -1: "bbx1 is occluded by bbx2"
}

def test():
    # bbx1 occludes bbx2
    bbx1 = np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
    bbx2 = np.array([[0, 2], [1, 2], [1, 3], [0, 3]]) + np.array([[1, 0]])
    flag = occlusion_query_between_bbx1_and_bbx2(bbx1, bbx2)
    print(occlusion_message[flag])
    # no occlusion
    bbx1 = np.array([[1, -1/2], [2, -1/2], [2, 1/2], [1, 1/2]])
    bbx2 = bbx1 + np.array([[0, 2]])
    flag = occlusion_query_between_bbx1_and_bbx2(bbx1, bbx2)
    print(occlusion_message[flag])


if __name__ == "__main__":
    test()