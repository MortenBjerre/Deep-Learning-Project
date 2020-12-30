import numpy as np
import torch
import torch.nn as nn
from utils import make_env

def evaluate_policy(policy, num_levels, num_envs=32):
    '''
    Evaluation function. Let's `num_envs` play until they are all death.
    Once an environments dies it's reward will no longer be counted.
    The function returns the average reward across the `num_envs` and the 
    std as well. 
    Slightly changed from the ipynb version with num_levels as parameters
    to accomodate seperate files.
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
    # Calculate average return
    total_reward = torch.stack(total_reward).sum(0)
    average_reward = total_reward.mean(0)
    average_reward_std = total_reward.std()

    return average_reward.item(), average_reward_std.item()

def evaluate_policy_update(policy, num_levels, num_envs=32, lives_per_environment = 5):
    '''
    Updated evaluation method where each environments has lives equal 
    to `lives_per_environment`. 

    Not used as evaluation method.
    '''
    results = []
    for i in range(lives_per_environment):
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

#r = evaluate_policy(policy, 1000, num_envs = 32, lives_per_environment = 5)
#k = r.mean(axis=0)
#print(k.mean())
#print(k.std())