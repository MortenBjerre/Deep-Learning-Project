# Hyperparameters
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
#os.chdir(r'/content/drive/My Drive/Colab Notebooks/Deep learning/Project')

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from torch.distributions import Categorical
import numpy as np
import time
import pandas as pd 
import time

def GPU_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
GPU_setup()

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

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

import math
def f(x):
    '''
    Used to calculate the number of output features from each IMPALA block.
    '''
    x = math.sqrt(x)
    x = math.ceil(x/2)
    return x**2

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class NatureDQNEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        #input : 64 x 64 x 3 (pixel x pixel x rgb)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
            nn.ReLU(), # 15 x 15 x 32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
            nn.ReLU(), # 6 x 6 x 64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(), # 4 x 4 x 64 = 1024
            Flatten(),
            nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)

class ImpalaEncoder_2(nn.Module):
    '''
    Used to test the expanded version of the impala encoder where
    the channels have been scaled by a factor of 2 compared to the standard. 
    '''
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=32)
        self.block2 = ImpalaBlock(in_channels=32, out_channels=64)
        self.block3 = ImpalaBlock(in_channels=64, out_channels=64)
        self.fc = nn.Linear(in_features = 64 * 8 * 8, out_features = feature_dim)
        self.relu = nn.ReLU()
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

class ImpalaEncoder_3(nn.Module):
    '''
    Used to test the expanded version of the impala encoder where
    the channels have been scaled by a factor of 2 compared to the standard
    and an additional fully connected layer was added. 
    '''
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=32)
        self.block2 = ImpalaBlock(in_channels=32, out_channels=64)
        self.block3 = ImpalaBlock(in_channels=64, out_channels=64)
        self.fc1 = nn.Linear(in_features = 64 * 8 * 8, out_features = 2048)
        self.fc2 = nn.Linear(in_features = 2048, out_features = feature_dim)
        self.relu = nn.ReLU()
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = Flatten()(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    '''
    Used in the IMPALA blocks.
    '''
    def __init__(self, in_channels):
        super().__init__() # <-
        self.convolution1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.convolution2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.convolution1(self.relu(x))
        output = self.convolution2(self.relu(output))
        return output + x

class ImpalaBlock(nn.Module):
    '''
    Used in the IMPALA encoder. The IMPALA encoder will use three of these
    blocks with varying in- and out-channels.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(out_channels)
        self.residual_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.convolution(x)
        x = self.maxpool(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        return x


class ImpalaEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features = feature_dim)
        self.relu = nn.ReLU()
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

class Policy(nn.Module):
    '''
    Policy class used to map from screen input to action distribution and
    value function. 
    '''
    def __init__(self, encoder, feature_dim, num_actions): 
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        self.ppo_loss = PPO_loss()
        self.value_loss = clipped_value_loss()

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = Categorical(logits=logits)

        return dist, value

# Define environment
env = make_env(num_envs, num_levels = num_levels)
# num_levels defines the number of levels the AI can train on
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
#encoder = Encoder(3, 1024)
encoder = NatureDQNEncoder(3, 1024)
policy = Policy(encoder, 1024, env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
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

"""# Evaluation method"""

def evaluate_policy(policy, num_envs=32):
    '''
    Evaluation function. Let's `num_envs` play until they are all death.
    Once an environments dies it's reward will no longer be counted.
    The function returns the average reward across the `num_envs` and the 
    std as well. 
    '''
    # Make evaluation environment
    eval_env = make_env(num_envs, start_level = num_levels, num_levels = num_levels) 
    obs = eval_env.reset()
    total_reward = []
    # Evaluate policy
    policy.eval()

    live_envs = np.array([True] * num_envs)
    while sum(live_envs) != 0: # runs until all environments are done
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

        t += 1
    # Calculate average return
    total_reward = torch.stack(total_reward).sum(0)
    average_reward = total_reward.mean(0)
    average_reward_std = total_reward.std()

    return average_reward.item(), average_reward_std.item()

"""# Training loop"""

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

    #evaluation during training 
    if c % 100 == 0: # done appr. every 800k step
        e_score, e_std = evaluate_policy(policy)
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
#save policy
torch.save({
    'policy_state_dict':policy.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict()
    },'checkpoint.pt')

"""## Saving the training data

We mainly used the HPC on DTU to train the AI but it was saved in the following way.
"""

folder_name = time.ctime()
folder_path = r'/content/drive/My Drive/Colab Notebooks/Deep learning/Project/Data/'
os.mkdir((folder_path + folder_name))

import imageio

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels) # 1 i stedet for num_envs 
obs = eval_env.reset()

frames = []
total_reward = []
# Evaluate policy
policy.eval()

live_envs = np.array([True] * num_envs)
t = 0
while sum(live_envs) != 0 or t > 20000: # runs until all environments are done or 20k steps
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

    t += 1

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

"""## Re evaluate training results 
"""

#path = '/content/drive/MyDrive/Project/Data/random'
os.chdir(path)
for i, folder in enumerate(os.listdir(path)):
    #print(folder)
    # Define network
    #encoder = ImpalaEncoder(3, 1024)

    encoder = ImpalaEncoder(3, 1024)
    policy = Policy(encoder, 1024, env.action_space.n)
    policy.cuda()

    # Define optimizer
    # these are reasonable values but probably not optimal
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

    # Define temporary storage
    # we use this to collect transitions during each iteration
    storage = Storage(
        env.observation_space.shape,
        num_steps,
        num_envs,
        gamma = 1.0
    )

    # Load from previous checkpoint
    checkpoint = torch.load(folder + '/checkpoint.pt')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #policy.eval()
    print('Policy loaded from checkpoint!')


    # Make evaluation environment
    hyperparams = pd.read_csv(folder + '/hyperparams.csv')

    eval_env = make_env(num_envs, start_level=hyperparams.num_levels.item(), 
                        num_levels=hyperparams.num_levels.item()) 
    obs = eval_env.reset()

    #frames = []
    total_reward = []
    # Evaluate policy
    policy.eval()

    live_envs = np.array([True] * num_envs)

    while sum(live_envs) != 0: # runs until all environments are done
        # Use policy
        action, log_prob, value = policy.act(obs)

        # Take step in environment
        obs, reward, done, info = eval_env.step(action)
        reward = [x['reward'] for x in info] #unnormalized reward
        reward = reward * live_envs # don't count reward if environment is done
        # update done environments
        level_done = np.array([x['prev_level_complete'] for x in info])

        # done is also true if the level is complete, so in that case we do not
        # want the ai to stop playing but continue with the next level
        live_envs = (np.invert(done) | level_done)  * live_envs 

        total_reward.append(torch.Tensor(reward))

        # Render environment and store
        #frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
        #frames.append(frame)

    # Calculate average return
    total_reward = torch.stack(total_reward).sum(0)
    average_reward = total_reward.mean(0)
    print('Average return:', average_reward.item())

    hyperparams['eval_average_reward'] = average_reward.item()
    hyperparams['eval_std'] =  total_reward.std().item()
    hyperparams.to_csv(folder + '/hyperparams.csv', index = False)

"""# Video

Block for making the video of the AI playing the game
"""

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels = num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
#encoder = Encoder(3, 1024)
encoder = ImpalaEncoder(3, 1024)
policy = Policy(encoder, 1024, env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma = 1.0
)

create_policy_from_checkpoint = True
os.chdir(r'/content/drive/MyDrive/Project/Data/impala_eps/Mon Dec 28 03%3A51%3A33 2020')
# Load from previous checkpoint
if 'checkpoint.pt' in os.listdir() and create_policy_from_checkpoint:
    print('Checkpoint in folder.')
    checkpoint = torch.load(os.getcwd() + '/checkpoint.pt')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #policy.eval()
    print('Policy loaded from checkpoint!')

import imageio
from random import randint

# Make evaluation environment
eval_env = make_env(1, start_level=1000, num_levels=1000, seed = randint(1,1000)) 
obs = eval_env.reset()
frames = []
policy.eval()

t=0
while t < 1500: # play for a fixed number of steps
    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)
    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
    frames.append(frame)
    t += 1

os.chdir(r'/content/drive/MyDrive/Project')
imageio.mimsave('best_policy.mp4', frames, fps=25)

"""# New evaluation method test
"""

# Define environment
# check the utils.py file for info on arguments
eval_env = make_env(32, start_level=num_levels, num_levels=num_levels) 
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
#encoder = Encoder(3, 1024)
encoder = ImpalaEncoder(3, 1024)
policy = Policy(encoder, 1024, env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma = 1.0
)

create_policy_from_checkpoint = True
os.chdir(r'/content/drive/MyDrive/Project/Data/data_impala/Fri Dec 11 08%3A52%3A13 2020')
# Load from previous checkpoint
if 'checkpoint.pt' in os.listdir() and create_policy_from_checkpoint:
    print('Checkpoint in folder.')
    checkpoint = torch.load(os.getcwd() + '/checkpoint.pt')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #policy.eval()
    print('Policy loaded from checkpoint!')

def evaluate_policy(policy, num_levels, num_envs=32, lives_per_environment = 5):
    results = []
    for i in range(lives_per_environment):
        print(i)
        total_reward = []
        eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels) 
        obs = eval_env.reset()
        live_envs = np.array([True] * num_envs)
        while sum(live_envs) != 0: # runs until all environments are done 
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

        # Calculate average return
        total_reward = torch.stack(total_reward).sum(0)
        results.append(total_reward.tolist())
    return np.array(results) #, average_reward.item(), average_reward_std.item()

r = evaluate_policy(policy, 1000, num_envs = 32, lives_per_environment = 5)
k = r.mean(axis=0)
print(k.mean())
print(k.std())