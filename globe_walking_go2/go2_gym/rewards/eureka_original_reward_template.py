import torch
import numpy as np
from globe_walking_go2.go2_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    
# INSERT EUREKA REWARD HERE

    def compute_fitness_score(self):
        return torch.ones_like(self.env.base_pos[:, 2])