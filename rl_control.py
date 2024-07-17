import numpy as np
import time
import math

DETERMINISTIC_EVALUATION = True


def point_traj(start_state, lookv_goal, spm_env, model, seed, logfilename=None):
    if logfilename is not None:
        spm_env.set_logfilename(logfilename)
    spm_env.reset()
    spm_env.seed(seed)
    if not spm_env.set_source(start_state[0], start_state[1], start_state[2]):
        raise RuntimeError("Input angles are in deep singularity")
    spm_env.set_dest_lookv_goal(lookv_goal)
    obs = np.array(spm_env.state, dtype=np.float32)
    traj, actions, sing_dict = [], [], {'singA': [], 'singB': []}
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
        action, _ = model.predict(obs, deterministic=DETERMINISTIC_EVALUATION)
        time_measures.append(time.perf_counter() - start_time)
        obs, reward, done, info = spm_env.step(action)
        step = step + 1
        actions.append(action)
        sing_dict['singA'].append(spm_env.S2s)
        sing_dict['singB'].append(min(spm_env.S1v))
        traj.append(spm_env.lookv)
        euler['theta'].append(spm_env.tetas)
        euler['phi'].append(spm_env.phis)
        euler['psi'].append(spm_env.psis)

        if done or step >= spm_env.max_steps:
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
                          ", Total prediction time: " + str(round(sum(time_measures), 4)) +
                          ", Initial Angle: " + str(round(math.degrees(init_angle), 3)) +
                          ", Final Angle: " + str(round(math.degrees(info['goal_ang']), 3)) +
                          ", Delta Angle: " + str(delta_theta) +
                          ", End reason: " + done_info +
                          ", Total singularity: " + str(total_singularity))
            print(summary)
            break

    return traj, actions, time_measures, sing_dict, obs, euler, done_info
