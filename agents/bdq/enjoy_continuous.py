import gym
import numpy as np
import os, sys
import time

path_agent_parent_dir = '../'  
sys.path.append(path_agent_parent_dir + '../')
sys.path.append(os.path.dirname('bdq') + path_agent_parent_dir)
path_logs = path_agent_parent_dir + 'bdq/' 

import envs
from bdq import deepq

# Enter environment name and numb sub-actions per joint 
env_name = 'Reacher6DOF-v0' ; num_actions_pad = 33 # ensure it's set correctly to the value used during training   

# Uncomment the pre-trained model that you wish to run
# Reacher3DOF-v0:
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_00-49-32_Reacher3DOF-v0.pkl' 
# Reacher4DOF-v0:
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_00-50-30_Reacher4DOF-v0.pkl' 
# Reacher5DOF-v0:
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_00-51-17_Reacher5DOF-v0.pkl' 
# Reacher6DOF-v0:
model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_01-03-10_Reacher6DOF-v0.pkl' 
# Reacher-v1: 
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_01-01-07_Reacher-v1.pkl' 
# Hopper-v1:
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_11-02-56_Hopper-v1.pkl' 
# Walker-v1:
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-02-01_13-08-52_Walker2d-v1.pkl' 
# Humanoid-v1:
#model_file_name = 'Branching_Dueling-reduceLocalMean_TD-target-mean_TD-errors-aggregation-v2_granularity-33_2018-01-29_11-52-42_Humanoid-v1.pkl' 


model_dir = '{}/trained_models/{}'.format(os.path.abspath(path_logs), env_name)

def main():
    env = gym.make(env_name)
    act = deepq.load("{}/{}".format(model_dir, model_file_name))
    
    num_action_dims = env.action_space.shape[0] 
    num_action_streams = num_action_dims
    num_actions = num_actions_pad*num_action_streams
    low = env.action_space.low 
    high = env.action_space.high 
    actions_range = np.subtract(high, low) 

    total_rewards = 0
    for i in range(100):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.02)
            
            action_idx = np.array(act(np.array(obs)[None], stochastic=False))
            action = action_idx / (num_actions_pad-1) * actions_range - high

            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print('Episode reward', episode_rew)
        total_rewards += episode_rew

    print('Mean episode reward: {}'.format(total_rewards/100))

if __name__ == '__main__':
    main()