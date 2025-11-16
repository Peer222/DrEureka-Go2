import torch
import numpy as np
from globe_walking.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    
    def _reward_height(self):
        env = self.env
        # Desired height is twice the ball radius
        desired_height = 2.0 * env.ball_radius
    
        # Reward when base stays above the threshold
        height_diff = env.base_pos[:, 2] - desired_height
        # Continuous reward using a sigmoid to provide a smooth
        # transition near the boundary.
        reward = torch.sigmoid(height_diff * 10.0)  # 10 scales transition steepness
    
        # Scale to give high priority to height maintenance
        return 1.0 * reward
    
    # ==============================================================
    # Velocity Penalty (keep robot stationary on the ball)
    # ==============================================================
    def _reward_velocity(self):
        env = self.env
        # Horizontal velocity components (local frame)
        # Discard vertical component to avoid lifting penalties.
        horiz_vel = env.base_lin_vel[:, 0:2]
        vel_norm_sq = torch.sum(horiz_vel ** 2, dim=1)
    
        # Penalise squared velocity (negative reward)
        return -0.1 * vel_norm_sq
    
    # ==============================================================
    # Action Regularisation (magnitude)
    # ==============================================================
    def _reward_action_magnitude(self):
        env = self.env
        action_norm_sq = torch.sum(env.actions ** 2, dim=1)
        return -0.05 * action_norm_sq
    
    # ==============================================================
    # Action Regularisation (rate of change)
    # ==============================================================
    def _reward_action_rate(self):
        env = self.env
        action_diff = env.actions - env.last_actions
        action_rate_norm_sq = torch.sum(action_diff ** 2, dim=1)
        return -0.05 * action_rate_norm_sq
    
    # ==============================================================
    # Torque Penalty
    # ==============================================================
    def _reward_torque(self):
        env = self.env
        torque_norm_sq = torch.sum(env.torques ** 2, dim=1)
        # The torque tensor is large; use a very small multiplier
        return -1e-05 * torque_norm_sq
    
    # ==============================================================
    # Contact Force Penalty
    # ==============================================================
    def _reward_contact_forces(self):
        env = self.env
        # Use only the forces on foot bodies
        foot_forces = env.contact_forces[:, env.feet_indices, :]
        # L2 norm of each foot's contact force
        foot_force_norm = torch.norm(foot_forces, dim=2)
        # Sum across all feet
        total_force = torch.sum(foot_force_norm, dim=1)
        # Penalise large contact forces
        return -0.01 * total_force

    # Success criteria as episode length
    def compute_success(self):
        return torch.ones_like(self.env.base_pos[:, 2])
