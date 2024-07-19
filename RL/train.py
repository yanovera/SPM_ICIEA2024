"""
This script trains a reinforcement learning (RL) model using the SPMEnv environment, which simulates a Spherical Parallel Manipulator (SPM).

Script Overview:
1. Imports and Setup:
   - Imports necessary libraries like gym, numpy, and stable_baselines3.
   - Defines constants and parameters for the model and training process.

2. Environment and Wrapper:
   - CustomWrapper: A wrapper for the SPMEnv environment to manage state, rewards, and steps.
   - make_env: A function to create multiple instances of the environment for parallel processing.

3. Training and Evaluation:
   - evaluate: A function to evaluate the trained model over a specified number of episodes.
   - main: The main function that sets up the environment, loads or initializes the model, and starts the training process.

Parameters Explained:
- PREFIX: Prefix for the model name.
- ALG: Algorithm used for training (e.g., A2C).
- N_PROC: Number of processes for parallel environment instances.
- MODEL_DIR: Directory to save the trained model.
- NET_ARCH_A2C: Network architecture for the A2C algorithm.
- NET_ARCH_TD3: Network architecture for the TD3 algorithm.
- LEARNING_RATE_A2C: Learning rate for the A2C algorithm.
- LEARNING_RATE_TD3: Learning rate for the TD3 algorithm.
- GAMMA: Discount factor for future rewards.
- LEARNING_STEPS: Total number of training steps.
- EVAL_FREQ: Frequency of evaluation during training.
- N_EVAL_EPISODES: Number of episodes for each evaluation.
- EVALUATION_EPISODES: Total number of episodes for final evaluation.
- DETERMINISTIC_EVALUATION: Whether to use deterministic actions during evaluation.
- SEED: Random seed for reproducibility.
- DEVICE: Device to run the model on (e.g., cuda for GPU).
"""

import math
import os
import time

import gym
import numpy as np
from stable_baselines3 import A2C, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment import SPM_Env_ver4 as SPM_Env

# Model Prefix
PREFIX = "DISCRETE_A2C_55"

#  Trained model, directory and file
ALG = A2C
N_PROC = 18

MODEL_DIR = "./models/" + PREFIX

# Learning parameters
NET_ARCH_A2C = [{'pi': [64, 64], 'vf': [64, 64]}]
NET_ARCH_TD3 = [400, 300]
LEARNING_RATE_A2C = 7e-4
LEARNING_RATE_TD3 = 1e-3
GAMMA = 0.99

LEARNING_STEPS = 1000000000000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 3000

EVALUATION_EPISODES = 10000  # Number of episodes for model evaluation
DETERMINISTIC_EVALUATION = True

SEED = 0

DEVICE = 'cuda'


# Define an environment function
# The multiprocessing implementation requires a function that can be called inside the process to instantiate a gym env
def make_env(rank, seed=SEED):
    """
    Utility function for multiprocessed env.
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = CustomWrapper(SPM_Env.SpmEnv())
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


#  Wrapper
class CustomWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        """
        Call the parent constructor, so we can access self.env later
        """
        super(CustomWrapper, self).__init__(env)
        self.prev_goal_distance = 0
        self.prev_singA = 0
        self.step_counter = 0
        self.done_info = ''

    def reset(self, random=True, random_dest=True, phis=0, tetas=0, psis=0, phis_dest=0, tetas_dest=0, psis_dest=0):
        """
        Reset the environment
        """
        obs = self.env.reset()
        self.prev_goal_distance = 0
        self.prev_singA = 0
        self.step_counter = 0
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional information
        """
        self.done_info = ''
        self.prev_goal_distance = self.env.goal_distance
        self.prev_singA = self.env.S2s
        try:
            obs, reward, done, info = self.env.step(action)
        except RuntimeError:
            # raise RuntimeError("deep singularity, increase number of samples")
            obs = self.env.state
            reward = -10
            done = True
            info = {"goal_ang": self.env.los_distance,
                    "singA": 0, "singB": 0}

        goal_distance_delta = self.prev_goal_distance - self.env.goal_distance
        singA_delta = info['singA'] - self.prev_singA

        if reward == self.env.reward_success:
            self.done_info = "Success"
            reward = self.env.par.reward_success
            done = True

        elif reward == self.env.reward_singul:
            self.done_info = "Singularity point"
            reward = self.env.par.reward_singularity
            done = True

        else:
            delta_step_reward = goal_distance_delta * self.env.par.reward_step_dist_factor
            delta_singA_reward = singA_delta * self.env.par.reward_step_singA
            reward = reward + self.env.par.reward_extra_step + delta_step_reward + delta_singA_reward
            if self.step_counter >= self.max_steps:
                self.done_info = "Out of steps"
                done = True

        self.step_counter += 1

        info["done_info"] = self.done_info

        return obs, reward, done, info


#  Evaluating a trained model
def evaluate(model, env, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param env: (gym.Env) the single evaluation environment
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    all_episode_steps = []
    all_episode_min_singA = []
    all_episode_min_singB = []

    success_counter = 0
    out_of_steps_counter = 0
    singularity_point_counter = 0
    prediction_time_counter = 0

    steps_counter = 0  # summing the steps
    step_max_ep = 0  # max number of steps per episode
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        # init_obs, _, _, _ = env.step([-1])
        init_angle = env.los_distance
        step = 0
        min_singA = np.inf
        min_singB = np.inf
        while not done:
            # _states are only useful when using LSTM policies
            start = time.perf_counter()
            action, _states = model.predict(obs, deterministic=DETERMINISTIC_EVALUATION)
            prediction_time_counter += time.perf_counter() - start
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            if info['singA'] < min_singA:
                min_singA = info['singA']
            if min(info['singB']) < min_singB:
                min_singB = min(info['singB'])

            step += 1

        print_info(init_angle, info['goal_ang'], i, step, sum(episode_rewards), env.done_info, min_singA, min_singB)
        all_episode_rewards.append(sum(episode_rewards))
        all_episode_min_singA.append(min_singA)
        all_episode_min_singB.append(min_singB)
        steps_counter += step
        if env.done_info == 'Success':
            success_counter += 1
            all_episode_steps.append(step)
            if step > step_max_ep:
                step_max_ep = step
        if env.done_info == 'Out of steps':
            out_of_steps_counter += 1
        if env.done_info == 'Singularity point':
            singularity_point_counter += 1

    print("Success percentage:", (success_counter/num_episodes) * 100, "%")
    if success_counter > 0:
        print("Average steps to success:", np.mean(all_episode_steps))
        print("Std. of steps to success", np.std(all_episode_steps))
    print("Maximum steps to success:", step_max_ep)

    print("Number of |success:", success_counter, "|",
          "Out of steps:", out_of_steps_counter, "|", "Singularity point:", singularity_point_counter, "|",)
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Trained Model Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    print("Minimum Sing A:", min(all_episode_min_singA))
    print("Minimum Sing B:", min(all_episode_min_singB))
    print("Average prediction time:", prediction_time_counter/steps_counter)

    return mean_episode_reward


def print_info(init_angle, goal_angle, episode, step, total_reward, done_info, min_singA, min_singB):
    init_ang_deg = math.degrees(init_angle)
    goal_ang_deg = math.degrees(goal_angle)
    delta_theta = round(init_ang_deg - goal_ang_deg, 3)
    print("Episode: " + str(episode) + ", Steps: " + str(
        step) + ", End reason " + str(done_info) + ", Total reward " + str(round(total_reward, 4)) +
          ", Initial Angle " + str(round(init_ang_deg, 3)) +
          ", Final Angle " + str(round(goal_ang_deg, 3)) +
          ", Delta Angle " + str(delta_theta) +
          ", Min. Sing A " + str(min_singA) + ", Min. Sing B " + str(min_singB))


def main():
    # Create save dir
    os.makedirs(MODEL_DIR, exist_ok=True)

    if N_PROC == 1:
        env = DummyVecEnv([lambda: CustomWrapper(SPM_Env.SpmEnv())])
    else:
        # env = SubprocVecEnv([make_env(i) for i in range(N_PROC)], start_method='spawn')
        env = SubprocVecEnv([make_env(i) for i in range(N_PROC)], start_method='fork')

    try:
        trained_model = ALG.load(MODEL_DIR + "/" + "best_model", env=env, verbose=1, device=DEVICE)
    except FileNotFoundError:
        print('WARNING!!! No saved model found.')
        if ALG == A2C:
            trained_model = ALG('MlpPolicy', env, device=DEVICE, verbose=1, learning_rate=LEARNING_RATE_A2C, gamma=GAMMA, policy_kwargs={'net_arch': NET_ARCH_A2C})
        elif ALG == TD3:
            trained_model = ALG('MlpPolicy', env, device=DEVICE, verbose=1, learning_rate=LEARNING_RATE_TD3, gamma=GAMMA, policy_kwargs={'net_arch': NET_ARCH_TD3}, train_freq=(20, "step"))
        else:
            raise RuntimeError("Unsupported training algorithm")

    # We will create one environment to evaluate the agent on
    eval_env = CustomWrapper(SPM_Env.SpmEnv(seed=SEED))
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
    trained_model.learn(total_timesteps=LEARNING_STEPS, callback=eval_callback)

    _ = evaluate(trained_model, eval_env, num_episodes=EVALUATION_EPISODES)

    env.close()


if __name__ == '__main__':
    main()
