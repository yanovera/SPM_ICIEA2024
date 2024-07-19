"""
This script evaluates trained reinforcement learning (RL) models against a classic control method for a Spherical Parallel Manipulator (SPM) and visualizes the results.

Script Overview:
1. Imports and Setup:
   - Imports necessary libraries like gym, numpy, pandas, matplotlib, plotly, and stable_baselines3.
   - Defines constants and parameters for evaluation and visualization.

2. Environment and Models:
   - Initializes the SPM environment and loads trained RL models (A2C and TD3).
   - Defines classic control parameters and initializes the classic control method.

3. Evaluation and Visualization:
   - Solver class: Encapsulates the solving method for each algorithm.
   - main: The main function that evaluates each algorithm, collects results, and visualizes trajectories and performance metrics.

Parameters Explained:
- PERFORM_A2C: Whether to evaluate the A2C model.
- PERFORM_TD3: Whether to evaluate the TD3 model.
- PERFORM_CLASSIC: Whether to evaluate the classic control method.
- START_STATE: Initial state of the SPM.
- GOAL_LOOKV: Goal look vector for the SPM.
- PLOT_EULER_2D: Whether to plot 2D Euler angles.
- SHOW_SINGULARITY_MAP: Whether to show the singularity map.
- PLOT_EULER_3D: Whether to plot 3D Euler angles.
- PLOT_THETAS: Whether to plot joint angles.
- ITERATIONS: Number of evaluation iterations.
- SEED: Random seed for reproducibility.
- ACTION_NOISE_SIGMA: Standard deviation of action noise.
- MEASUREMENT_NOISE_SIGMA: Standard deviation of measurement noise.
- DEVICE: Device to run the models on (e.g., cpu or cuda).
- RL_MODELS_PATH: Path to the directory containing trained RL models.
- LOG_PATH: Path to the directory for saving logs.
- SINGULARITY_MAP_FILE: Path to the singularity map file.
- A2C_MODEL_NAME: Name of the A2C model.
- A2C_MODEL_PARAMS: Parameters for the A2C model.
- TD3_MODEL_NAME: Name of the TD3 model.
- TD3_MODEL_PARAMS: Parameters for the TD3 model.
- CLASSIC_PARAMS: Parameters for the classic control method.
"""


import io
import math
import pickle
from base64 import b64encode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots
from stable_baselines3 import PPO, TD3

import classic_control
import rl_control
from RL.enums import ActionsSpace
from RL.parameters_ver4 import Parameters
from environment import SPM_Env_ver4 as SPM_Env

buffer = io.StringIO()

PERFORM_A2C = True
PERFORM_TD3 = True
PERFORM_CLASSIC = True
PERFORM_ROLL_AND_PITCH_GRID_SEARCH = False
PERFORM_JOINTS_ELEVATION_GRID_SEARCH = False

START_STATE = [ 0.449, -0.447,  0.491]
GOAL_LOOKV = [0.602, 0.079, 0.794]
PLOT_EULER_2D = True
SHOW_SINGULARITY_MAP = True
PLOT_EULER_3D = True
PLOT_THETAS = True

ITERATIONS = 1
SEED = 10

ACTION_NOISE_SIGMA = 0  # rad/sec
MEASUREMENT_NOISE_SIGMA = 0  # rad

DEVICE = 'cpu'

RL_MODELS_PATH = './RL/models/'
LOG_PATH = './logs/'
SINGULARITY_MAP_FILE = 'grids/singularity_map.pickle'

A2C_MODEL_NAME = 'FREE_DISCRETE_A2C_55'
A2C_MODEL_PARAMS = Parameters(actions_type=ActionsSpace.DISCRETE, ang_threshold=0.2,
                              log_flag=True, enable_azimuth_control=True, action_noise_sigma=ACTION_NOISE_SIGMA,
                              measurement_noise_sigma=MEASUREMENT_NOISE_SIGMA)

TD3_MODEL_NAME = 'FREE_CONT_TD3_55'
TD3_MODEL_PARAMS = Parameters(actions_type=ActionsSpace.CONTINUOUS, ang_threshold=0.2,
                              log_flag=True, enable_azimuth_control=True, action_noise_sigma=ACTION_NOISE_SIGMA,
                              measurement_noise_sigma=MEASUREMENT_NOISE_SIGMA)


CLASSIC_PARAMS = Parameters(classic_control_K=250, enable_azimuth_control=False,
                            log_flag=True, max_steps=10000, action_noise_sigma=ACTION_NOISE_SIGMA,
                            measurement_noise_sigma=MEASUREMENT_NOISE_SIGMA)


class Solver:
    def __init__(self, name, solve_method, symbol, color):
        self.name = name
        self.solve = solve_method
        self.symbol = symbol
        self.color = color

    def solve(self, start_state, goal_lookv):
        return self.solve(start_state, goal_lookv)


def create_download(fig, port):
    fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="graph", figure=fig),
        html.A(
            html.Button("Download HTML"),
            id="download",
            href="data:text/html;base64," + encoded,
            download="plotly_graph.html"
        )
    ])
    app.run_server(port=port)


def main():
    env = SPM_Env.SpmEnv()
    env.seed(SEED)
    info_df = None
    algorithms = []
    # load networks
    if PERFORM_A2C:
        a2c_env = SPM_Env.SpmEnv(parameters=A2C_MODEL_PARAMS)
        a2c_env.seed(SEED)
        a2c_model = PPO.load(RL_MODELS_PATH + A2C_MODEL_NAME + "/best_model", env=a2c_env, verbose=1, device=DEVICE)

        algorithms += [Solver("A2C",
                       lambda start_state,
                              goal_lookv,
                              logfilename: rl_control.point_traj(start_state,
                                                                 goal_lookv,
                                                                 spm_env=a2c_env,
                                                                 model=a2c_model,
                                                                 seed=seed,
                                                                 logfilename=logfilename),
                              symbol='circle', color='blue')]

    if PERFORM_TD3:
        td3_env = SPM_Env.SpmEnv(parameters=TD3_MODEL_PARAMS)
        td3_env.seed(SEED)
        td3_model = TD3.load(RL_MODELS_PATH + TD3_MODEL_NAME + "/best_model", env=td3_env, verbose=1, device=DEVICE)
        algorithms += [Solver("TD3",
                              lambda start_state,
                                     goal_lookv,
                                     logfilename: rl_control.point_traj(start_state,
                                                                        goal_lookv,
                                                                        spm_env=td3_env,
                                                                        model=td3_model,
                                                                        seed=seed,
                                                                        logfilename=logfilename),
                              symbol='circle-open', color='orange')]

    if PERFORM_CLASSIC:
        classic_env = SPM_Env.SpmEnv(parameters=CLASSIC_PARAMS)
        classic_env.seed(SEED)
        algorithms += [Solver("UF",
                              lambda start_state,
                                     goal_lookv,
                                     logfilename: classic_control.point_traj(start_state,
                                                                             goal_lookv,
                                                                             spm_env=classic_env,
                                                                             seed=seed,
                                                                             logfilename=logfilename),
                              symbol='cross', color='green')]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        specs=[[{"type": "table"}], [{"type": "scatter"}]])

    seed = SEED
    for i in range(ITERATIONS):
        print('Iteration', i+1, 'out of ', ITERATIONS, "; seed = ", seed)
        env.seed(seed)
        _ = env.reset()
        start_state = np.array([env.phis, env.tetas, env.psis])
        start_lookv = env.lookv             # random
        delta_theta = math.degrees(env.los_distance)
        goal_lookv = env.lookv_goal  # random or [0, 0, 1] (homing)

        euler_dict = {}

        if START_STATE is not None:
            start_state = START_STATE
            start_lookv = np.dot(SPM_Env.Q321(x1=start_state[0], x2=start_state[1], x3=start_state[2]), env.homing)
        if GOAL_LOOKV is not None:
            goal_lookv = GOAL_LOOKV

        for algorithm in algorithms:
            print('Executing: ', algorithm.name)
            try:
                traj, actions, total_time, sing_dict, last_state, euler, done_info = \
                    algorithm.solve(start_state, goal_lookv, LOG_PATH+algorithm.name+"_"+str(seed)+".txt")
                df = pd.DataFrame(traj, columns=['x', 'y', 'z'])
                df['i'] = list(range(len(df)))
                df['actions'] = actions
                df['A singularity'] = sing_dict['singA']
                df['B singularity'] = sing_dict['singB']
                df['angular distance'] = [0] + [np.arccos(np.dot(traj[i], traj[i+1])) for i in range(len(traj)-1)]
                fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], marker=dict(size=5, color=df['A singularity'], colorscale=[(0,"red"), (1,"blue")], cmin=0.05, cmax=0.25, symbol=algorithm.symbol), mode='lines+markers',
                                           text=df['A singularity'], name=algorithm.name, connectgaps=True, line={'color':algorithm.color, 'width': 5}))
                new_info_row = pd.DataFrame.from_dict(
                    {'Algorithm': [algorithm.name], 'Steps': [len(traj)], 'Avg_Time': [sum(total_time)/len(traj)], 'Phi_f': [last_state[0]],
                     'Theta_f': [last_state[1]], 'Psi_f': [last_state[2]], 'singA_avg': np.average(sing_dict['singA']), 'Traj. length': df['angular distance'].sum(), 'End': [done_info]})
            except RuntimeError:
                new_info_row = pd.DataFrame.from_dict(
                    {'Algorithm': [algorithm.name], 'Steps': [0], 'Avg_Time': [0],
                     'Phi_f': [0], 'Theta_f': [0], 'Psi_f': [0], 'singA_avg': [0], 'Traj. length': 0,
                     'End': ['Failure']})
                euler = {'phi': [], 'theta': [], 'psi': []}
                df = pd.DataFrame()
            if info_df is None:
                info_df = new_info_row
            else:
                info_df = pd.concat([info_df, new_info_row], ignore_index=True)

            euler_dict[algorithm.name] = (euler['phi'], euler['theta'], euler['psi'])

        seed += 1

        if ITERATIONS == 1:
            if info_df is not None:
                fig.update_layout(height=900, showlegend=True, title_font_size=25, title={
                    'text': f"SPM Trajectories<br>"
                            f"<sup>"
                            f"Start position: [φ={start_state[0]:.3f}, θ={start_state[1]:.3f}, ψ={start_state[2]:.3f}], "
                            f"Goal vector [{goal_lookv[0]:.3f}, {goal_lookv[1]:.3f}, {goal_lookv[2]:.3f}], "
                            f"LOS distance: {delta_theta:.2f}\N{DEGREE SIGN}</sup>",
                            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, legend=dict(font=dict(size=22, color="black")))


                fig.add_trace(go.Table(header=dict(values=["Algorithm", "Steps", "Avg. Control Time", "Final φ", "Final θ", "Final ψ", "Average |det(A)|", "Traj. Length", "End reason"]),
                                       cells=dict(values=[info_df[k].tolist() for k in info_df.columns[0:]], format=["", "", ".3", ".6", ".6", ".6", ".6", ".4", ""])),
                                       row=1, col=1)

                fig.add_trace(go.Scatter3d(x=[goal_lookv[0]], y=[goal_lookv[1]], z=[goal_lookv[2]], mode='markers', name='End'))
                fig.add_trace(go.Scatter3d(x=[start_lookv[0]], y=[start_lookv[1]], z=[start_lookv[2]], mode='markers', name='Start'))

                # Add space search
                df_space = pd.read_csv('grids/WorkSpace_60deg_Sing_0_05.csv')
                df_space['color'] = pd.Series([-0.1 for x in range(len(df.index))])
                fig.add_trace(go.Scatter3d(x=df_space['x'], y=df_space['y'], z=df_space['z'], marker=dict(size=2), mode='markers',
                                                   text=df_space['color'], name="space"))

                create_download(fig, 9021)
            if PLOT_EULER_2D:
                fig = plt.figure()
                for alg_name, euler_angles in euler_dict.items():
                    plt.scatter(euler_angles[1], euler_angles[0], zorder=-1)
                    plt.plot(euler_angles[1], euler_angles[0], label=alg_name, linewidth=2)

                if SHOW_SINGULARITY_MAP:
                    with open(SINGULARITY_MAP_FILE, 'rb') as handle:
                        sing_map_dict = pickle.load(handle)
                        plt.pcolormesh(sing_map_dict['phis'], sing_map_dict['thetas'], sing_map_dict['singularity_map'], cmap='magma', zorder=0)

                plt.scatter(start_state[1], start_state[0], label='Start')
                multi_target_positions = SPM_Env.get_target_candidates(start_roll=start_state[0], start_pitch=start_state[1], lookv_goal=goal_lookv)
                target_thetas, target_phis = [], []
                for target_position in multi_target_positions:
                    target_thetas.append(target_position[1])
                    target_phis.append(target_position[0])
                target_thetas.append(multi_target_positions[0][1])
                target_phis.append(multi_target_positions[0][0])
                plt.plot(target_thetas, target_phis, label='End')
                plt.xlabel(r'$\theta$ [rad]', fontsize=12)
                plt.ylabel(r'$\phi$ [rad]', fontsize=12)
                plt.tick_params(axis='both', which='major', labelsize=12)
                plt.tick_params(axis='both', which='minor', labelsize=12)
                plt.legend(fontsize=12)

            if PLOT_EULER_3D:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                for alg_name, euler_angles in euler_dict.items():
                    # Create a scatter plot of the 3D points colored by their values
                    ax.scatter(euler_angles[0], euler_angles[1], euler_angles[2], s=7)
                    ax.plot(euler_angles[0], euler_angles[1], euler_angles[2], label=alg_name)

                ax.scatter(start_state[0], start_state[1], start_state[2], label='Start')
                multi_target_positions = SPM_Env.get_target_candidates(start_roll=start_state[0], start_pitch=start_state[1], lookv_goal=goal_lookv)
                target_psis, target_thetas, target_phis = [], [], []
                target_singAs = []
                for target_position in multi_target_positions:
                    target_phis.append(target_position[0])
                    target_thetas.append(target_position[1])
                    target_psis.append(target_position[2])
                    target_singAs.append(env.get_singularities(target_position)['singA'])
                ax.scatter(target_phis, target_thetas, target_psis, c=target_singAs, label='End')
                ax.set_xlabel('Roll')
                ax.set_ylabel('Pitch')
                ax.set_zlabel('Yaw')
                plt.legend()

            if PLOT_THETAS:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                thetas_dict = {}
                for alg_name, euler_angles in euler_dict.items():
                    positions = zip(euler_angles[0], euler_angles[1], euler_angles[2])
                    thetas_dict[alg_name] = []
                    for pos in positions:
                        Qm = SPM_Env.Q321(pos[0], pos[1], pos[2])
                        tetavIK, _, SingFlag = SPM_Env.SPMIK(Qm, env.vast, env.geopara)
                        if SingFlag == 0:
                            raise RuntimeError('invalid position found')
                        thetas_dict[alg_name].append(tetavIK)
                    # Extract coordinates and values from the dictionary
                    coords = np.array(thetas_dict[alg_name])

                    # Create a scatter plot of the 3D points colored by their values
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=7)
                    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], label=alg_name)

                # Add labels
                ax.set_xlabel('Theta1')
                ax.set_ylabel('Theta2')
                ax.set_zlabel('Theta3')
                plt.legend()

            plt.show()

    a2c_df = info_df[info_df["Algorithm"] == "A2C"]
    td3_df = info_df[info_df["Algorithm"] == "TD3"]
    cc_df = info_df[info_df["Algorithm"] == "UF"]

    success_rates = {"A2C": a2c_df[a2c_df["End"] == 'Success'].count()["Algorithm"]/ITERATIONS*100,
                     "TD3": td3_df[td3_df["End"] == 'Success'].count()["Algorithm"]/ITERATIONS*100,
                     "UF": cc_df[cc_df["End"] == 'Success'].count()["Algorithm"]/ITERATIONS*100,
                     }

    traj_length_mean = {"A2C": a2c_df["Traj. length"].mean(),
                        "TD3": td3_df["Traj. length"].mean(),
                        "UF": cc_df["Traj. length"].mean(),
                        }

    traj_length_std = {"A2C": a2c_df["Traj. length"].std(),
                       "TD3": td3_df["Traj. length"].std(),
                       "UF": cc_df["Traj. length"].std(),
                       }

    traj_singA_mean = {"A2C": a2c_df["singA_avg"].mean(),
                        "TD3": td3_df["singA_avg"].mean(),
                        "UF": cc_df["singA_avg"].mean(),
                        }

    print("Success rates[%]: ", success_rates)
    print("Traj. length mean: ", traj_length_mean)
    print("Traj. length std: ", traj_length_std)
    print("Traj. singA mean: ", traj_singA_mean)


if __name__ == '__main__':
    main()
