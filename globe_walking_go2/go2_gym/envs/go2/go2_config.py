from typing import Union

from params_proto import Meta

from globe_walking_go2.go2_gym.envs.base.legged_robot_config import Cfg


def config_go2(Cnfg: Union[Cfg, Meta]):
    Cnfg.robot.name = "go2"  # from train script
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

    _ = Cnfg.control
    _.control_type = "P"  #  'actuator_net'  # TODO go1 uses actuator net here!
    _.stiffness = {'joint': 20.}  # [N*m/rad]
    _.damping = {'joint': 0.5}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = [] # globe walking go 1 does not use contact termination on "base"
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    # ------------------ Go2: _.flip_visual_attachments = True
    # ------------------ Go1: _.flip_visual_attachments = False
    _.flip_visual_attachments = True
    _.fix_base_link = False
    _.num_actuated_dof = 12  # from go2.py
    _.use_vhacd = True  # from go2.py
    _.vhacd_resolution = 500000  # from go2.py

    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.use_terminal_body_height = True

    _ = Cnfg.env
    _.num_observations = 56
    _.num_observation_history = 15 # from train script
    # _.observe_vel = False
    # _.num_envs = 4000
    _.episode_length_s = 40

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    ### from train script
    _.exclusive_phase_offset = False
    _.balance_gait_distribution = False
    _.gaitwise_curricula = False
    ###

    # terrain configuration
    Cnfg.terrain.border_size = 0
    Cnfg.terrain.mesh_type = "boxes_tm"
    Cnfg.terrain.num_cols = 20
    Cnfg.terrain.num_rows = 20
    Cnfg.terrain.terrain_length = 5.0
    Cnfg.terrain.terrain_width = 5.0
    Cnfg.terrain.num_border_boxes = 5
    Cnfg.terrain.teleport_thresh = 0.3
    Cnfg.terrain.teleport_robots = False
    Cnfg.terrain.center_robots = False
    Cnfg.terrain.center_span = 3
    Cnfg.terrain.horizontal_scale = 0.05
    Cnfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cnfg.terrain.curriculum = False
    Cnfg.terrain.difficulty_scale = 1.0
    Cnfg.terrain.max_step_height = 0.26
    Cnfg.terrain.min_step_run = 0.25
    Cnfg.terrain.max_step_run = 0.4
    Cnfg.terrain.max_init_terrain_level = 1

    Cnfg.terrain.measure_heights = False
    Cnfg.terrain.terrain_noise_magnitude = 0.0
