import torch
import numpy as np
from globe_walking_go2.go2_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaRewardStub():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_fitness_score(self):
        return torch.ones_like(self.env.base_pos[:, 2])

    def compute_curriculum_score(self):
        return torch.ones_like(self.env.base_pos[:, 2])

class EurekaReward(EurekaRewardStub):

# INSERT EUREKA REWARD HERE
