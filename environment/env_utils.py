import numpy as np
# Functions


# Termination
"""
Checking if the the episode should terminate
3 cases of termination exist:
    Pointing vector reached his destination (the wanted orientation is obtained - success)
    Singularity A is equal or less than singularity threshold
    Singularity B is equal or less than singularity threshold
"""


def termf(singA, singB, lookv_goal_ang, ang_threshold, sing_threshold):
    if singA <= sing_threshold:
        return True
    elif singB[0] <= sing_threshold or singB[1] <= sing_threshold or singB[2] <= sing_threshold:
        return True
    elif lookv_goal_ang <= ang_threshold:
        return True
    else:
        return False


# Reward
"""
Calculating the reward
"""


def rewardf(singA, singB, lookv_goal_ang, ang_threshold, sing_threshold, reward_success,
            reward_sing):
    if singA <= sing_threshold:  # Singularity A
        return reward_sing
    elif singB[0] <= sing_threshold or singB[1] <= sing_threshold or singB[2] <= sing_threshold:  # Singularity B
        return reward_sing
    elif lookv_goal_ang <= ang_threshold:  # Success
        return reward_success
    else:  # No success and no Singularity
        reward = 0  # (singA + min(singB)) / 2
        return reward


# EA321
"""
Computes the Euler Angles Phi Theta Psi from the Rotation Operator Q
according to the sequence 321
y1 --- psi radians  [-pi,pi]
y2 --- teta   [-pi/2,pi/2]
y3 --- phi   [-pi,pi]
Notice: the OUTPUT yv displays the angles in the order Phi Teta Psi
"""


def EA321Q(x) -> np.array(float):
    r11s = x[0, 0]
    r21s = x[1, 0]
    r31s = x[2, 0]
    r32s = x[2, 1]
    r33s = x[2, 2]
    r12s = x[0, 1]
    r22s = x[1, 1]

    dums = r11s ** 2 + r21s ** 2

    if abs(r31s) != 1:
        y2 = np.arctan2(-r31s, np.sqrt(dums))
        y1 = np.arctan2(r21s, r11s)
        y3 = np.arctan2(r32s, r33s)
    elif r31s == -1:
        y2 = np.pi / 2
        y1 = 0
        y3 = np.arctan2(r12s, r22s)
    elif r31s == 1:
        y2 = -np.pi / 2
        y1 = 0
        y3 = -np.arctan2(r12s, r22s)
    else:
        y1, y2, y3 = 0, 0, 0
        print('error in the input of the EA321Q function')
    yv = np.array([y3, y2, y1])
    return yv


# Q321
"""
Computes the Rotation Operator Q, from the Euler Angles Phi Theta Psi the according to 321 sequence
x1 = phi
x2 = teta
x3 = psi 
"""


def Q321(x1, x2, x3):
    c1 = np.cos(x1)
    s1 = np.sin(x1)
    c2 = np.cos(x2)
    s2 = np.sin(x2)
    c3 = np.cos(x3)
    s3 = np.sin(x3)

    d1m = np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]])
    d2m = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    d3m = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    y = np.matmul(np.matmul(d3m, d2m), d1m)
    return y


# crossVm
def crossVM(x):
    if x.shape == (3,):
        y = np.array([[0, - x[2], x[1]], [x[2], 0, - x[0]], [- x[1], x[0], 0]])
    elif x.shape == (3, 3):
        y = np.array([x[3][2]], x[1][3], x[2][1])
    else:
        y = 0
        print('error in the input dimensions ')
    return y


# SPMIK
"""
SPM Inverse kinematics
Given the rotation matrix Q, calculate the joint angles 3x1 vector tetav
"""


def SPMIK(Qm, vast, geopara, all_solutions: bool = False):
    SingFlag = 1
    if all_solutions:
        discriminant_sign = np.array([-1, 1])
    else:
        discriminant_sign = -1
    a1s, a2s = geopara[0], geopara[1]
    b1s, b2s = geopara[2], geopara[3]
    se1s, se2s, se3s = geopara[4], geopara[5], geopara[6]
    ce1s, ce2s, ce3s = geopara[7], geopara[8], geopara[9]

    v1ast = vast[0]
    v2ast = vast[1]
    v3ast = vast[2]

    v1v = np.dot(Qm, v1ast)
    v2v = np.dot(Qm, v2ast)
    v3v = np.dot(Qm, v3ast)

    #  Joint 1

    A1s = np.sin(a1s - b1s) * (se1s * v1v[0] + ce1s * v1v[1]) + np.cos(a1s - b1s) * v1v[2] + np.cos(a2s)
    B1s = np.sin(a1s) * (ce1s * v1v[0] - se1s * v1v[1])
    C1s = -np.sin(a1s + b1s) * (se1s * v1v[0] + ce1s * v1v[1]) + np.cos(a1s + b1s) * v1v[2] + np.cos(a2s)
    D1s = B1s ** 2 - A1s * C1s

    if D1s < 0:  # Singularity check
        return 0, 0, 0

    # T11s = (-B1s+np.sqrt(D1s))/A1s
    # teta11s = 2*np.arctan(T11s)
    T12s = (-B1s + discriminant_sign * np.sqrt(D1s)) / A1s
    teta12s = 2 * np.arctan(T12s)

    #  Joint 2

    A2s = np.sin(a1s - b1s) * (se2s * v2v[0] + ce2s * v2v[1]) + np.cos(a1s - b1s) * v2v[2] + np.cos(a2s)
    B2s = np.sin(a1s) * (ce2s * v2v[0] - se2s * v2v[1])
    C2s = -np.sin(a1s + b1s) * (se2s * v2v[0] + ce2s * v2v[1]) + np.cos(a1s + b1s) * v2v[2] + np.cos(a2s)
    D2s = B2s ** 2 - A2s * C2s

    if D2s < 0:  # Singularity check
        return 0, 0, 0

    T22s = (-B2s + discriminant_sign * np.sqrt(D2s)) / A2s
    teta22s = 2 * np.arctan(T22s)

    #  Joint 3

    A3s = np.sin(a1s - b1s) * (se3s * v3v[0] + ce3s * v3v[1]) + np.cos(a1s - b1s) * v3v[2] + np.cos(a2s)
    B3s = np.sin(a1s) * (ce3s * v3v[0] - se3s * v3v[1])
    C3s = -np.sin(a1s + b1s) * (se3s * v3v[0] + ce3s * v3v[1]) + np.cos(a1s + b1s) * v3v[2] + np.cos(a2s)
    D3s = B3s ** 2 - A3s * C3s

    if D3s < 0:  # Singularity check
        return 0, 0, 0

    T32s = (-B3s + discriminant_sign * np.sqrt(D3s)) / A3s
    teta32s = 2 * np.arctan(T32s)

    if all_solutions:
        tetavIK = np.array(np.meshgrid(teta12s, teta22s, teta32s)).T.reshape(-1, 3)
    else:
        tetavIK = np.array([teta12s, teta22s, teta32s])  # tetaijs i-arm index j-sign of discriminant (j=1-> + , j=2 -> -)
    Ds = np.array([D1s, D2s, D3s])

    return tetavIK, Ds, SingFlag


# w - vectors calculation
def w123v(geopara, tetavIK):
    teta1s = tetavIK[0]
    teta2s = tetavIK[1]
    teta3s = tetavIK[2]

    a1s, a2s = geopara[0], geopara[1]
    b1s, b2s = geopara[2], geopara[3]
    se1s, se2s, se3s = geopara[4], geopara[5], geopara[6]
    ce1s, ce2s, ce3s = geopara[7], geopara[8], geopara[9]
    sa1s = np.sin(a1s)
    ca1s = np.cos(a1s)
    sb1s = np.sin(b1s)
    cb1s = np.cos(b1s)

    w1v = np.array([se1s * (sb1s * ca1s + cb1s * sa1s * np.cos(teta1s))
                    - ce1s * np.sin(teta1s) * sa1s,
                    ce1s * (sb1s * ca1s + cb1s * sa1s * np.cos(teta1s))
                    + se1s * np.sin(teta1s) * sa1s,
                    sa1s * sb1s * np.cos(teta1s) - ca1s * cb1s])

    w2v = np.array([se2s * (sb1s * ca1s + cb1s * sa1s * np.cos(teta2s))
                    - ce2s * np.sin(teta2s) * sa1s,
                    ce2s * (sb1s * ca1s + cb1s * sa1s * np.cos(teta2s))
                    + se2s * np.sin(teta2s) * sa1s,
                    sa1s * sb1s * np.cos(teta2s) - ca1s * cb1s])

    w3v = np.array([se3s * (sb1s * ca1s + cb1s * sa1s * np.cos(teta3s))
                    - ce3s * np.sin(teta3s) * sa1s,
                    ce3s * (sb1s * ca1s + cb1s * sa1s * np.cos(teta3s))
                    + se3s * np.sin(teta3s) * sa1s,
                    sa1s * sb1s * np.cos(teta3s) - ca1s * cb1s])

    return np.array([w1v, w2v, w3v])


# v - vectors calculation
"""rotation of vector v star to v in world frame"""


def v123v(Qm, vast):
    v1v = np.dot(Qm, vast[0])
    v2v = np.dot(Qm, vast[1])
    v3v = np.dot(Qm, vast[2])
    return np.array([v1v, v2v, v3v])


def azimuth_angle(source: np.array, target: np.array) -> float:
    """
    returns ccw angle from source to target horizontal projections
    """
    a = source[:2]
    b = target[:2]
    normalizer = np.linalg.norm(a) * np.linalg.norm(b)
    if normalizer == 0:
        return 0
    cosTh = np.dot(a, b) / normalizer
    sinTh = np.cross(a, b) / normalizer
    return np.arctan2(sinTh, cosTh)


def get_target_candidates(start_roll, start_pitch, lookv_goal, step=0.1):
    target_candidates = []
    for yaw in np.arange(-np.pi, np.pi, step=step):
        target_candidates.append(get_target_position(start_roll, start_pitch, yaw, lookv_goal))
    return target_candidates


def get_target_position(start_roll, start_pitch, start_yaw, lookv_goal):
    """Compute target position using rotation matrices."""
    initial_orientation = Q321(start_roll, start_pitch, start_yaw)
    lookv = np.dot(initial_orientation, np.array([0, 0, 1]))

    R = rotation_matrix_from_vectors(lookv, lookv_goal)
    final_orientation = np.matmul(R, initial_orientation)

    target_roll, target_pitch, target_yaw = euler_from_matrix(final_orientation)
    return target_roll, target_pitch, target_yaw


def rotation_matrix_from_vectors(a, b):
    """Compute the rotation matrix that rotates vector a to align with vector b."""
    v = np.cross(a, b)
    c = np.dot(a, b)
    I = np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = I + vx + np.matmul(vx, vx) * (1 / (1 + c))
    return R


def euler_from_matrix(matrix):
    """Extract Euler angles from a rotation matrix."""
    pitch = np.arcsin(-matrix[2, 0])
    if np.cos(pitch) != 0:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
    return roll, pitch, yaw


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_discrete_oper_actions(tetadotv0: [list[float], float]):
    # Operators rotating space to be multiplied by tetadot
    oper_actions = np.array([[1, 1, -1], [1, 1, 0],
                            [1, -1, 1], [1, -1, -1], [1, -1, 0],
                            [1, 0, 1], [1, 0, -1], [1, 0, 0],
                            [-1, 1, 1], [-1, 1, -1], [-1, 1, 0],
                            [-1, -1, 1], [-1, -1, 0],
                            [-1, 0, 1], [-1, 0, -1], [-1, 0, 0],
                            [0, 1, 1], [0, 1, -1], [0, 1, 0],
                            [0, -1, 1], [0, -1, -1], [0, -1, 0],
                            [0, 0, 1], [0, 0, -1],
                            ])
    if isinstance(tetadotv0, np.ndarray):
        oper_actions = np.reshape([oper_actions * oper_speed for oper_speed in tetadotv0],
                                  (len(oper_actions) * len(tetadotv0), 3))
    else:
        oper_actions = oper_actions * tetadotv0
    return oper_actions


def get_design_params():
    # GEOMETRICAL PARAMETERS [rad]
    a1s = np.deg2rad(65)
    a2s = np.deg2rad(60)
    b1s = np.deg2rad(0)
    sb1s = np.sin(b1s)
    cb1s = np.cos(b1s)
    b2s = np.deg2rad(110)
    sb2s = np.sin(b2s)
    cb2s = np.cos(b2s)
    psi0s = np.deg2rad(0)
    eta1s = np.deg2rad(0)
    se1s = np.sin(eta1s)
    ce1s = np.cos(eta1s)
    eta2s = np.deg2rad(240)
    se2s = np.sin(eta2s)
    ce2s = np.cos(eta2s)
    eta3s = np.deg2rad(120)
    se3s = np.sin(eta3s)
    ce3s = np.cos(eta3s)

    geopara = np.array([a1s, a2s, b1s, b2s, se1s, se2s, se3s, ce1s, ce2s, ce3s])

    # u vectors
    u1v = np.array([-sb1s * se1s, sb1s * ce1s, -cb1s])
    u2v = np.array([-sb1s * se2s, sb1s * ce2s, -cb1s])
    u3v = np.array([-sb1s * se3s, sb1s * ce3s, -cb1s])

    # v star - v vector in reference state (home)
    v1ast = np.array([sb2s * np.sin(eta1s - psi0s), sb2s * np.cos(eta1s - psi0s),
                           cb2s])
    v2ast = np.array([sb2s * np.sin(eta2s - psi0s), sb2s * np.cos(eta2s - psi0s),
                           cb2s])
    v3ast = np.array([sb2s * np.sin(eta3s - psi0s), sb2s * np.cos(eta3s - psi0s),
                           cb2s])

    uvectors = np.array([u1v, u2v, u3v])

    vast = np.array([v1ast, v2ast, v3ast])

    return geopara, uvectors, vast
