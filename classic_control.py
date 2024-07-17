import numpy as np
import time
import math
from environment.SPM_Env_ver4 import SPMIK, w123v, v123v, Q321


def control(state, lookv_goal, vast, geopara, u1v, u2v, u3v, K=1):
    Qm = Q321(state[0], state[1], state[2])

    tetavIK, _, SingFlag = SPMIK(Qm, vast, geopara)  # IK calculation of operators rotation
    if SingFlag == 0:  # Checking if the mechanism is in deep singularity, meaning rotation matrix is complex
        raise RuntimeError("something went wrong, deep singularity has been reached")

    # w1v, v1v, w2v, v2v, w3v, v3v = 0, 0, 0, 0, 0, 0
    # w - vectors calculation
    w1v, w2v, w3v = w123v(geopara, tetavIK)

    #  v - vectors calculation
    v1v, v2v, v3v = v123v(Qm, vast)

    S1v = abs(np.array([np.dot(np.cross(w1v, u1v), v1v), np.dot(np.cross(w2v, u2v), v2v),
                             np.dot(np.cross(w3v, u3v), v3v)]))

    am = np.array([np.cross(w1v, v1v), np.cross(w2v, v2v), np.cross(w3v, v3v)])
    bm = np.diag(S1v)

    JSDKm = np.matmul(np.linalg.inv(am), bm)  # !! erase the MINUS - sign if IK uses the + ++ solution
    JSIKm = np.linalg.inv(JSDKm)

    lookv = np.dot(Qm, np.array([0, 0, 1]))
    q = rotation_quaternion(lookv, lookv_goal)

    omega_desired = K * q[1:]

    tetadotv = np.dot(JSIKm, omega_desired)

    return tetadotv


def rotation_quaternion(a, b):
    # Normalize the vectors
    a_unit = a / np.linalg.norm(a)
    b_unit = b / np.linalg.norm(b)

    # Compute the cross product (rotation axis)
    v = np.cross(a_unit, b_unit)

    # Compute the dot product (cosine of angle)
    cos_theta = np.dot(a_unit, b_unit)

    # Compute the angle of rotation
    theta = np.arccos(cos_theta)

    # Compute the quaternion
    w = np.cos(theta / 2)
    xyz = v * np.sin(theta / 2)

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def point_traj(start_state, lookv_goal, spm_env, seed, logfilename=None):
    spm_env.par.classic_control = True
    if logfilename is not None:
        spm_env.set_logfilename(logfilename)
    spm_env.reset()
    spm_env.seed(seed)
    if not spm_env.set_source(start_state[0], start_state[1], start_state[2]):
        raise RuntimeError("Input angles are in deep singularity")
    spm_env.set_dest_lookv_goal(lookv_goal)
    obs = np.array(spm_env.state, dtype=np.float32)
    total_rewards, traj, actions, sing_dict = [], [], [], {'singA': [], 'singB': []}
    euler = {'theta': [], 'phi': [], 'psi': []}
    init_angle = spm_env.los_distance
    step = 0
    time_measures = [0]
    traj.append(spm_env.lookv)
    actions.append(0)
    sing_dict['singA'].append(spm_env.S2s)
    sing_dict['singB'].append(min(spm_env.S1v))
    euler['theta'].append(spm_env.tetas)
    euler['phi'].append(spm_env.phis)
    euler['psi'].append(spm_env.psis)
    while True:
        start_time = time.perf_counter()
        k = spm_env.par.classic_control_K/spm_env.los_distance
        action = control(state=obs, lookv_goal=spm_env.lookv_goal, vast=spm_env.vast, geopara=spm_env.geopara, u1v=spm_env.uvectors[0], u2v=spm_env.uvectors[1], u3v=spm_env.uvectors[2], K=k)
        time_measures.append(time.perf_counter() - start_time)
        obs, reward, done, info = spm_env.step(action)
        total_rewards.append(reward)
        step = step + 1
        actions.append(action)
        sing_dict['singA'].append(spm_env.S2s)
        sing_dict['singB'].append(min(spm_env.S1v))
        traj.append(spm_env.lookv)
        euler['theta'].append(spm_env.tetas)
        euler['phi'].append(spm_env.phis)
        euler['psi'].append(spm_env.psis)

        if done:
            if reward == spm_env.reward_success:
                done_info = "Success"
            elif reward == spm_env.reward_singul:
                done_info = "Singularity point"
            elif step >= spm_env.max_steps:
                done_info = "Out of steps"
            else:
                done_info = "Unknown reason"
            delta_theta = round(math.degrees(init_angle) - math.degrees(info['goal_ang']), 3)
            total_singularity = round((sum(sing_dict['singA']) + sum(sing_dict['singB'])) / 2, 3)
            summary = str("Steps: " + str(step) +
                          ", Total reward: " + str(round(sum(total_rewards), 4)) +
                          ", Total prediction time: " + str(round(sum(time_measures), 3)) +
                          ", Initial Angle: " + str(round(math.degrees(init_angle), 3)) +
                          ", Final Angle: " + str(round(math.degrees(info['goal_ang']), 3)) +
                          ", Delta Angle: " + str(delta_theta) +
                          ", End reason: " + done_info +
                          ", Total singularity: " + str(total_singularity))
            print(summary)
            break

    return traj, actions, time_measures, sing_dict, obs, euler, done_info
