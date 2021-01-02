# procgen should be installed using the following command
#!pip install procgen

# Hyperparameters
# optimal hyperparameters
total_steps = 8e6
num_envs = 32
num_levels = 1000
num_steps = 256 
num_epochs = 3
batch_size = 1024
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
learning_rate = 4e-4
optimizer_eps = 1e-7

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from torch.distributions import Categorical
import numpy as np
import pandas as pd 
import time, math, imageio
from PPO import PPO_loss, clipped_value_loss
from naturedqnencoder import NatureEncoder
from impala import ImpalaEncoder
from policy import Policy
from evaluation_method import evaluate_policy

def GPU_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
GPU_setup()

# Define environment
env = make_env(num_envs, num_levels = num_levels)

# num_levels defines the number of levels the AI can train on
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
#encoder = NatureEncoder(3, 1024)
encoder = ImpalaEncoder(3, 1024)
policy = Policy(encoder, 1024, env.action_space.n)
policy.cuda()

# Define optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=optimizer_eps)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma = 1.0
)


create_policy_from_checkpoint = False
# Load from previous checkpoint
if 'checkpoint.pt' in os.listdir() and create_policy_from_checkpoint:
    print('Checkpoint in folder.')
    checkpoint = torch.load(os.getcwd() + '/checkpoint.pt')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Policy loaded from checkpoint!')

# Run training
obs = env.reset()
step = 0
std, step_list, mean_reward = [], [], []
penalty_for_dying = 10
step_list_eval = []
evaluation_score = []
evaluation_score_std = []
c = 0
start_training_time = time.time()

while step < total_steps:
    # Use policy to collect data for num_steps steps
    policy.eval()

    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        #reward = reward - done * penalty_for_dying - used in an experiment
                
        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)
        
        # Update current observation
        obs = next_obs

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()
    
    # Optimize policy
    policy.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)
        for batch in generator:
            b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

            # Get current policy outputs
            new_dist, new_value = policy(b_obs)
            new_log_prob = new_dist.log_prob(b_action)

            # Clipped policy objective
            pi_loss = policy.ppo_loss(new_log_prob, b_log_prob, b_advantage, eps)

            # Clipped value function objective
            value_loss = policy.value_loss(new_value, b_value, b_returns, eps)

            # Entropy loss
            entropy_loss = new_dist.entropy().mean()

            # Backpropagate losses
            loss =  pi_loss + value_coef * value_loss - entropy_coef * entropy_loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

            # Update policy
            optimizer.step()
            optimizer.zero_grad()

    #evaluation during training to see progress
    if c % 100 == 0: # done appr. every 800k step
        e_score, e_std = evaluate_policy(policy, num_levels)
        evaluation_score.append(e_score)
        evaluation_score_std.append(e_std)
        step_list_eval.append(step)
    c += 1

    # Update stats
    step += num_envs * num_steps

    step_list.append(step)
    mean_reward.append(storage.get_reward().item())
    std.append(storage.reward.sum(0).std().item())
    print(f'Step: {step}\tMean reward: {storage.get_reward()}')


print('Completed training!')
end_training_time = time.time()

# the following is used to save the data, policy and create a video
folder_name = time.ctime()
folder_path = r'/content/drive/My Drive/Colab Notebooks/Deep learning/Project/Data/'
os.mkdir((folder_path + folder_name))

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels) # 1 i stedet for num_envs 
obs = eval_env.reset()

frames = []
total_reward = []
# Evaluate policy
policy.eval()

live_envs = np.array([True] * num_envs)
while sum(live_envs) != 0: # runs until all environments are done or 20k steps
    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)
    
    reward = [x['reward'] for x in info] #unnormalized reward
    reward = reward * live_envs # don't count reward if environment is done
    # update done environments
    level_done = np.array( [x['prev_level_complete'] for x in info] )

    # done is also true if the level is complete, so in that case we do not
    # want the ai to stop playing but continue with the next level
    live_envs = (np.invert(done) | level_done)  * live_envs 

    total_reward.append(torch.Tensor(reward))

    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
    frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0)
average_reward = total_reward.mean(0)
print('Average return:', average_reward)
imageio.mimsave('vid.mp4', frames, fps=25)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave((folder_path + folder_name)+'/vid.mp4', frames, fps=25)

# Save training in csv

comment = """
tester lige hvordan vi evaluerer
"""
d = {'total_steps'          : total_steps,
      'num_envs'            : num_envs,
      'num_levels'          : num_levels,
      'num_steps'           : num_steps,
      'num_epochs'          : num_epochs,
      'batch_size'          : batch_size,
      'eps'                 : eps,
      'grad_eps'            : grad_eps,
      'value_coef'          : value_coef,
      'entropy_coef'        : entropy_coef,
      'learning_rate'       : learning_rate,
      'optimizer_eps'       : optimizer_eps,
      'eval_average_reward' : average_reward.item(),
      'eval_std'            : total_reward.std().item(),
      'training_time_min'   : (end_training_time - start_training_time)/60,
      'encoder'             : 'NatureDQN',
      'comments'            : comment}
hyperparams = pd.DataFrame([d])

hyperparams.to_csv((folder_path + folder_name) + '/hyperparameters.csv', index = False)

d = {'step': step_list,
     'std_reward': std,
     'mean_reward': mean_reward}
data = pd.DataFrame(d)
data.to_csv((folder_path + folder_name) + '/data.csv', index = False)


d = {'step': step_list_eval,
     'std_reward': evaluation_score_std,
     'mean_reward': evaluation_score}

data_evaluation = pd.DataFrame(d)
data_evaluation.to_csv((folder_path + folder_name) + '/eval_training.csv', index = False)

torch.save({
    'policy_state_dict':policy.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict()
    }, (folder_path + folder_name) + '/checkpoint.pt')