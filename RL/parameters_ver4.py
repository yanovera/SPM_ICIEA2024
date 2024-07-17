from RL.enums import ActionsSpace
from pydantic import BaseModel

default_actions_type = ActionsSpace.DISCRETE
default_classic_control = False
default_classic_control_K = 1
default_enable_azimuth_control = False  # use only for evaluation
default_use_grid = True  # if true, grid is used for filtering unreachable sources & destinations
default_roll_and_pitch_grid_file = 'grids/euler_grid_1.pkl'  # applicable when target_mode != JOINTS_ELEVATION


############Maximum operator steps limit###################
default_max_steps = 400  # actions operator steps limit

############Rewards##################
default_reward_step_dist_factor = 30  # Multiplies the angular difference from target, resulting a positive reward when the difference is decreased
default_reward_step_singA = 10  # Multiplies the difference in singA, resulting a positive reward when the singA is increased
default_reward_success = 50  # Signals the RL trainer that the agent has reached the target vector within the 'ang_threshold'
default_reward_singularity = -70  # Signals the RL trainer that the agent has reached singularity
default_reward_extra_step = -0.2  # minus for extra step

############Environment############
default_sing_threshold = 0.05  # Singularity threshold. Threshold decreases -> LOS cone angle increase, control decreases
default_ang_threshold = 0.2  # Sets the precision of the destination [deg]
default_tetadotv0 = [100, 200, 400]  # Sets the size of a single operator step (deg/sec)
default_yaw_speed = 5000  # (deg/sec)
default_action_noise_sigma = 0.0  # sigma for noise normal distribution
default_measurement_noise_sigma = 0.0  # sigma for noise normal distribution

############Resetting Angles top platform euler angles Boundings############
# Maximum and minimum values that will shuffle while Resetting the source vector
default_init_look_v_min = 0  # [deg].
default_init_look_v_max = 55  # [deg].

# Maximum and minimum values that will shuffle while Resetting the destination vectors
default_goal_look_v_min = 0  # [deg].
default_goal_look_v_max = 55  # [deg].

################Time Deltas##############
default_timestep = 0.001
default_sample_time = 0.001

############Log###################
default_log_flag = False
default_log_filename = "./logs/logfile.txt"


############Play############
# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
# path_model = os.path.join(script_dir, r"models/20211212-145311/Episode_4600")

class Parameters(BaseModel):
    actions_type = default_actions_type
    max_steps = default_max_steps
    reward_step_dist_factor = default_reward_step_dist_factor
    reward_success = default_reward_success
    reward_singularity = default_reward_singularity
    reward_step_singA = default_reward_step_singA
    reward_extra_step = default_reward_extra_step
    sing_threshold = default_sing_threshold
    ang_threshold = default_ang_threshold
    tetadotv0 = default_tetadotv0
    yaw_speed = default_yaw_speed
    action_noise_sigma = default_action_noise_sigma
    measurement_noise_sigma = default_measurement_noise_sigma
    init_look_v_min = default_init_look_v_min
    init_look_v_max = default_init_look_v_max
    goal_look_v_min = default_goal_look_v_min
    goal_look_v_max = default_goal_look_v_max
    log_flag = default_log_flag
    log_filename = default_log_filename
    classic_control = default_classic_control
    classic_control_K = default_classic_control_K
    enable_azimuth_control = default_enable_azimuth_control
    roll_and_pitch_grid_file = default_roll_and_pitch_grid_file
    use_grid = default_use_grid
    timestep = default_timestep
    sample_time = default_sample_time


