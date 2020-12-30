import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import exp

class PPO_loss():
    def __call__(self, new_log_pi, old_log_pi, advantage, epsilon: float):
        # compute ppo loss
        ratio = exp(new_log_pi - old_log_pi)
        clipped_ratio = ratio.clamp(min = 1 - epsilon, max = 1 + epsilon)
        reward = torch.min(ratio * advantage, clipped_ratio * advantage)

        return - reward.mean()


class clipped_value_loss():
    def __call__(self, new_value, old_value, old_return, epsilon: float):
        # compute clipped value loss
        clipped_value = old_value + (new_value - old_value).clamp(min = -epsilon, max = epsilon)
        value_function_loss = torch.max((new_value - old_return) ** 2, (clipped_value - old_return) ** 2)
        return 0.5 * (value_function_loss.mean())

