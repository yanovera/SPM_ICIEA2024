import os
import random
from math import degrees as deg  # Converting radians to degrees
from math import radians as rad  # Converting degrees to radians

import gym
from gym.vector.utils import spaces
from scipy.sparse.linalg import expm

from RL.parameters_ver4 import Parameters
from RL.enums import ActionsSpace

from typing import Optional
from environment.env_utils import *
from environment.GridMap import GridMap


class SpmEnv(gym.Env):
    def __init__(self, parameters: Optional[Parameters] = None, seed=None):
        if parameters is None:
            self.par = Parameters()
        else:
            self.par = parameters
        # PROCESS INITIALIZATION
        self.t_END = self.par.timestep  # time step (sec)
        self.dt = self.par.sample_time  # sample time (sec)
        self.tetadotv0 = np.deg2rad(self.par.tetadotv0)  # rd/sec
        self.yaw_speed = rad(self.par.yaw_speed)
        self.sing_threshold = self.par.sing_threshold  # singularity threshold
        self.ang_threshold = rad(self.par.ang_threshold)  # destination angle threshold [rad]
        self.rng = random.Random(seed)

        # DESIGN PARAMETERS [rad]
        self.geopara, self.uvectors, self.vast = get_design_params()

        # GRID INITIALIZATION
        self.grid_map = None
        if self.par.use_grid:
            self.grid_map = GridMap(filename=self.par.roll_and_pitch_grid_file, joints_mode=False)

        # Platform INITIALIZATION

        self.phis = rad(0)  # phi
        self.tetas = rad(0)  # teta
        self.psis = rad(0)  # psi
        self.Qm = Q321(self.phis, self.tetas, self.psis)  # Q matrix

        self.homing = np.array([0, 0, 1])  # homing vector
        self.lookv_goal = np.array([0, 0, 1])  # look vector destination
        self.elevation_goal = 0
        self.lookv = np.array([0, 0, 1])  # vector normal to platform
        self.los_distance = 0
        self.elevation_distance = 0
        self.azimuth_distance = 0
        self.roll_and_pitch_distance = 0
        self.goal_distance = 0
        self.elevation = 0

        # Platform Destination

        # v vector
        self.v1v = np.dot(self.Qm, self.vast[0])
        self.v2v = np.dot(self.Qm, self.vast[1])
        self.v3v = np.dot(self.Qm, self.vast[2])

        self.wtovang = np.array([0, 0, 0])

        # Singularities
        self.S1v = np.array([0, 0, 0])  # detB type of singularity (vector of 3 operators singularities)
        self.S2s = 0  # detA type of singularity (scalar)

        self.OperAction = None
        self.max_steps = self.par.max_steps
        if self.par.enable_azimuth_control:
            self.max_steps += int(np.pi / self.yaw_speed / self.t_END)
        self.set_actions()

        self.tetadotv = np.zeros(3)  # rd/sec

        # RL INITIALIZATION
        self.t_time = 0
        self.reward = 0
        self.state = np.array([])
        self.set_state()
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.state.shape[0],))
        self.numofobs = self.state.shape[0]  # Number of observations
        self.reward_success = 10
        self.reward_singul = -10

        # Log file
        self.logfilename = self.par.log_filename
        self.log_flag = self.par.log_flag

    def set_actions(self):
        if self.par.actions_type == ActionsSpace.DISCRETE:
            self.OperAction = get_discrete_oper_actions(self.tetadotv0)
            self.action_space = spaces.Discrete(self.OperAction.shape[0])
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(low=0, high=10000)
        self.rng.seed(seed)
        return [seed]

    def get_tetadotv(self, action):
        if self.par.classic_control:
            tetadotv = action
        else:
            if self.par.actions_type == ActionsSpace.DISCRETE:
                tetadotv = self.OperAction[action]  # rd/sec
            else:
                tetadotv = action * np.max(self.tetadotv0)
        if self.azimuth_distance != 0:
            elevation_reached = np.abs(self.elevation_goal - self.elevation) <= self.ang_threshold
            if elevation_reached:
                tetadotv = np.zeros(3)
            margin = self.yaw_speed - abs(max(tetadotv))
            rate = abs(self.azimuth_distance / self.t_END)
            tetadotv = tetadotv + np.ones(3) * np.sign(self.azimuth_distance) * min(margin, rate)
        action_noise = np.array([self.rng.gauss(mu=0, sigma=self.par.action_noise_sigma) for _ in range(3)])
        tetadotv += action_noise
        return tetadotv

    def step(self, action):
        self.t_time += self.t_END
        self.Qm = Q321(self.phis, self.tetas, self.psis)  # Q matrix
        self.tetadotv = self.get_tetadotv(action)

        tetavIK, _, SingFlag = SPMIK(self.Qm, self.vast, self.geopara)  # IK calculation of operators rotation
        if SingFlag == 0:  # Checking if the mechanism is in deep singularity, meaning rotation matrix is complex
            raise RuntimeError("something went wrong, deep singularity has been reached")

        # w1v, v1v, w2v, v2v, w3v, v3v = 0, 0, 0, 0, 0, 0
        # w - vectors calculation
        w1v, w2v, w3v = w123v(self.geopara, tetavIK)

        #  v - vectors calculation
        v1v, v2v, v3v = v123v(self.Qm, self.vast)

        am = np.array([np.cross(w1v, v1v), np.cross(w2v, v2v), np.cross(w3v, v3v)])
        bm = np.diag(self.S1v)

        k_END = int(self.t_END / self.dt)  # number of samples

        if self.los_distance > self.ang_threshold:
            #  Rotating the manipulator
            for _ in range(0, k_END):
                # Space Jacobian for Direct and Inverse Kinematics
                JSDKm = np.matmul(np.linalg.inv(am), bm)  # !! erase the MINUS - sign if IK uses the + ++ solution

                omSv = np.dot(JSDKm, self.tetadotv)  # omega - Angular velocity of the platform

                #  propagation of the Q matrix and of the joints angles
                self.Qm = np.dot(expm(np.dot(crossVM(omSv), self.dt)), self.Qm)

                if np.any(np.iscomplex(self.Qm)):  # Checking if the mechanism is in deep singularity
                    raise RuntimeError("something went wrong, deep singularity has been reached")

                tetavIK, _, SingFlag = SPMIK(self.Qm, self.vast, self.geopara)  # new operators angles (after propagating)
                if SingFlag == 0:  # Checking if the mechanism is in deep singularity
                    raise RuntimeError("something went wrong, deep singularity has been reached")

                # w - vectors calculation
                w1v, w2v, w3v = w123v(self.geopara, tetavIK)

                #  v - vectors calculation
                v1v, v2v, v3v = v123v(self.Qm, self.vast)

                #  A - matrix calculation
                am = np.array([np.cross(w1v, v1v), np.cross(w2v, v2v), np.cross(w3v, v3v)])
                self.S2s = abs(np.linalg.det(am))
                if self.S2s <= self.sing_threshold:
                    break
                #  B - matrix calculation
                self.S1v = abs(np.array([np.dot(np.cross(w1v, self.uvectors[0]), v1v), np.dot(np.cross(w2v, self.uvectors[1]), v2v),
                                         np.dot(np.cross(w3v, self.uvectors[2]), v3v)]))
                bm = np.diag(self.S1v)
                if self.S1v[0] <= self.sing_threshold or self.S1v[1] <= self.sing_threshold or \
                        self.S1v[2] <= self.sing_threshold:
                    break

        # Updating the new state
        self.phis, self.tetas, self.psis = EA321Q(self.Qm)  # Updating the euler angle of the platform
        self.set_state()

        # Reward function
        self.reward = rewardf(self.S2s, self.S1v, self.goal_distance, self.ang_threshold, self.sing_threshold,
                              self.reward_success, self.reward_singul)

        # Termination check
        done = termf(self.S2s, self.S1v, self.goal_distance, self.ang_threshold, self.sing_threshold)

        if self.log_flag:
            self.wtovang = np.array([np.arccos(np.dot(np.cross(self.uvectors[0], w1v) / np.linalg.norm(np.cross(self.uvectors[0], w1v)),
                                                      np.cross(w1v, v1v) / np.linalg.norm(np.cross(w1v, v1v)))),
                                     np.arccos(np.dot(np.cross(self.uvectors[1], w2v) / np.linalg.norm(np.cross(self.uvectors[1], w2v)),
                                                      np.cross(w2v, v2v) / np.linalg.norm(np.cross(w2v, v2v)))),
                                     np.arccos(np.dot(np.cross(self.uvectors[2], w3v) / np.linalg.norm(np.cross(self.uvectors[2], w3v)),
                                                      np.cross(w3v, v3v) / np.linalg.norm(np.cross(w3v, v3v))))])
            visualization_data = ("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f\n" %
                                  (deg(tetavIK[0]), deg(tetavIK[1]), deg(tetavIK[2]), deg(self.wtovang[0]),
                                   deg(self.wtovang[1]), deg(self.wtovang[2]), deg(self.phis), deg(self.tetas),
                                   deg(self.psis), self.S2s, self.S1v[0], self.S1v[1], self.S1v[2],
                                   deg(self.los_distance), self.reward, self.lookv[0], self.lookv[1], self.lookv[2], self.lookv_goal[0], self.lookv_goal[1], self.lookv_goal[2], self.los_distance, self.elevation_distance, self.azimuth_distance))
            with open(self.logfilename, 'a') as f:
                f.write(visualization_data)

        return np.array(self.state, dtype=np.float32), self.reward, done, {"goal_ang": self.los_distance,
                                                                "singA": self.S2s, "singB": self.S1v}

    def reset(self):
        if os.path.exists(self.logfilename):
            os.remove(self.logfilename)
        self.t_time = 0

        # Raffling initial state, 3 euler angles of the platform
        source_ok = False
        while not source_ok:
            phis = self.rng.uniform(-np.pi, np.pi)
            tetas = self.rng.uniform(-np.pi, np.pi)
            psis = self.rng.uniform(-np.pi, np.pi)
            look_v = np.dot(Q321(phis, tetas, psis), self.homing)
            look_v_angle = np.arccos(np.dot(look_v, self.homing))
            if look_v_angle < rad(self.par.init_look_v_min) or look_v_angle > rad(self.par.init_look_v_max):
                continue
            source_ok = self.set_source(phis=phis, tetas=tetas, psis=psis, check_connectivity=True)

        # Raffling destination state, looking vector
        phis_dest = 0.0
        tetas_dest = self.rng.uniform(rad(self.par.goal_look_v_min), rad(self.par.goal_look_v_max))
        psis_dest = self.rng.uniform(-np.pi, np.pi)
        Qm_dest = Q321(phis_dest, tetas_dest, psis_dest)  # Q matrix
        self.set_dest_lookv_goal(lookv_goal=np.dot(Qm_dest, self.homing))

        return np.array(self.state, dtype=np.float32)

    def set_source(self, phis=0.0, tetas=0.0, psis=0.0, check_connectivity=False):
        self.Qm = Q321(phis, tetas, psis)  # Q matrix
        self.phis, self.tetas, self.psis = EA321Q(self.Qm)  # fix ambiguity of euler angles
        if check_connectivity:
            label_check = self.grid_map.get_label(roll=self.phis, pitch=self.tetas)
            if label_check != self.grid_map.homing_label:
                return False
        tetavIK, _, SingFlag = SPMIK(self.Qm, self.vast, self.geopara)  # IK calculation of operators rotation
        if SingFlag == 0:  # Checking if the mechanism is in deep singularity, meaning rotation matrix is complex
            return False

        w1v, w2v, w3v = w123v(self.geopara, tetavIK)
        v1v, v2v, v3v = v123v(self.Qm, self.vast)

        #  A - matrix calculation
        am = np.array([np.cross(w1v, v1v), np.cross(w2v, v2v), np.cross(w3v, v3v)])
        self.S2s = abs(np.linalg.det(am))

        #  B - matrix calculation
        self.S1v = abs(np.array([np.dot(np.cross(w1v, self.uvectors[0]), v1v), np.dot(np.cross(w2v, self.uvectors[1]), v2v),
                                 np.dot(np.cross(w3v, self.uvectors[2]), v3v)]))

        reset_sing_check = min(self.S2s, self.S1v[0], self.S1v[1], self.S1v[2])
        if reset_sing_check <= self.sing_threshold:
            return False

        self.set_state()
        self.set_logfile(w1v, v1v, w2v, v2v, w3v, v3v, tetavIK)
        return True

    def set_logfilename(self, logfilename: str):
        self.logfilename = logfilename

    def set_dest_lookv_goal(self, lookv_goal):
        self.lookv_goal = lookv_goal / np.linalg.norm(lookv_goal)
        self.elevation_goal = np.arccos(np.dot(self.lookv_goal, np.array([0, 0, 1])))
        self.set_state()

    def set_state(self):
        self.lookv = np.dot(self.Qm, self.homing)
        self.los_distance = np.arccos(np.dot(self.lookv_goal, self.lookv))
        self.elevation = np.arccos(np.dot(self.lookv, np.array([0, 0, 1])))
        self.elevation_distance = np.abs(self.elevation_goal - self.elevation)
        if self.par.enable_azimuth_control:
            self.azimuth_distance = azimuth_angle(source=self.lookv, target=self.lookv_goal)
        tetavIK, _, _ = SPMIK(self.Qm, self.vast, self.geopara)
        measured_position = np.array([self.phis, self.tetas, self.psis]) + np.array([self.rng.gauss(mu=0, sigma=self.par.measurement_noise_sigma) for _ in range(3)])
        self.state = np.concatenate((measured_position, [self.elevation_goal]))
        self.set_goal_distance()

    def set_logfile(self, w1v, v1v, w2v, v2v, w3v, v3v, tetavIK):
        if self.log_flag:
            os.makedirs(os.path.dirname(self.logfilename), exist_ok=True)
            with open(self.logfilename, 'w') as f:
                self.wtovang = np.array(
                    [np.arccos(np.dot(np.cross(self.uvectors[0], w1v) / np.linalg.norm(np.cross(self.uvectors[0], w1v)),
                                      np.cross(w1v, v1v) / np.linalg.norm(np.cross(w1v, v1v)))),
                     np.arccos(np.dot(np.cross(self.uvectors[1], w2v) / np.linalg.norm(np.cross(self.uvectors[1], w2v)),
                                      np.cross(w2v, v2v) / np.linalg.norm(np.cross(w2v, v2v)))),
                     np.arccos(np.dot(np.cross(self.uvectors[2], w3v) / np.linalg.norm(np.cross(self.uvectors[2], w3v)),
                                      np.cross(w3v, v3v) / np.linalg.norm(np.cross(w3v, v3v))))])
                f.write("Parameters: "+self.par.json()+"\n")
                f.write("teta1, teta2, teta3, WtoV1, WtoV2, WtoV3, Plat_psi, Plat_teta, Plat_phi, "
                        "singA, singBleg1, singBleg2, singBleg3, AngleToDest, reward, lookv1, lookv2, lookv3, lookv_goal1, lookv_goal2, lookv_goal3, los distance, elevation error, azimuth error\n")
                f.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.2f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f\n" %
                        (deg(tetavIK[0]), deg(tetavIK[1]), deg(tetavIK[2]), deg(self.wtovang[0]),
                         deg(self.wtovang[1]), deg(self.wtovang[2]), deg(self.phis), deg(self.tetas),
                         deg(self.psis), self.S2s, self.S1v[0], self.S1v[1], self.S1v[2],
                         deg(self.los_distance), self.reward, self.lookv[0], self.lookv[1], self.lookv[2], self.lookv_goal[0], self.lookv_goal[1], self.lookv_goal[2], self.los_distance, self.elevation_distance, self.azimuth_distance))

    def get_singularities(self, state, absolute_values=True):
        phis, tetas, psis = state[:3]
        Qm = Q321(phis, tetas, psis)  # Q matrix
        tetavIK, _, SingFlag = SPMIK(Qm, self.vast, self.geopara)  # IK calculation of operators rotation

        if SingFlag == 0:  # Checking if the mechanism is in deep singularity, meaning rotation matrix is complex
            return {'singA': 0, 'singB': 0}

        w1v, w2v, w3v = w123v(self.geopara, tetavIK)
        v1v, v2v, v3v = v123v(Qm, self.vast)

        #  A - matrix calculation
        am = np.array([np.cross(w1v, v1v), np.cross(w2v, v2v), np.cross(w3v, v3v)])
        S2s = np.linalg.det(am)

        #  B - matrix calculation
        S1v = np.array([np.dot(np.cross(w1v, self.uvectors[0]), v1v), np.dot(np.cross(w2v, self.uvectors[1]), v2v),
                            np.dot(np.cross(w3v, self.uvectors[2]), v3v)])

        if absolute_values:
            return {'singA': abs(S2s), 'singB': np.min(abs(S1v))}
        else:
            return {'singA': S2s, 'singB': np.min(S1v)}

    def render(self, mode="human"):  # TODO
        """Renders the environment"""
        pass

    def close(self):  # TODO
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def set_goal_distance(self):
        if self.par.enable_azimuth_control or self.par.classic_control:
            self.goal_distance = self.los_distance
        else:
            self.goal_distance = self.elevation_distance
