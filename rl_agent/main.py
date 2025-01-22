import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DQN, PPO
import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.envs.registration import register
import glob
import numpy as np
import torch
from pathlib import Path
import json
from rl_agent.HumanoidEnv import HumanoidEnv  # Import your custom environment class


class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_dir="reward_logs", log_file_name="reward_log.json", verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file_path = os.path.join(log_dir, log_file_name)
        self.all_episode_logs = []  # This will store all episodes' data

        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_episode_end(self) -> None:
        # Get the info dictionary from the environment
        info = self.locals.get("infos", [])[0]
        episode_info = info.get('episode', {})

        if episode_info or 1 > 0:  # Always true, to avoid conditional logging issues
            # Extract the reward components
            reward_components = {key: episode_info[key] for key in episode_info.keys() if key not in ['r', 'l', 't']}
            
            # Prepare data to save
            log_data = {
                'total_reward': episode_info.get('r', None),
                'reward_components': reward_components,
                'episode_length': episode_info.get('l', None),
                'episode_time': episode_info.get('t', None),
                'full_info': info  # Save the entire info dictionary for debugging purposes
            }

            # Append the log data to the list
            self.all_episode_logs.append(log_data)

            # Save to a single JSON file
            with open(self.log_file_path, 'w') as f:
                json.dump(self.all_episode_logs, f, indent=4)



class VelocityLoggerCallback(BaseCallback):
    def __init__(self, velocity_filepath, verbose=0):
        super(VelocityLoggerCallback, self).__init__(verbose)
        self.velocity_filepath = velocity_filepath
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.velocity_filepath), exist_ok=True)

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [])
        if len(info) > 0 and "x_velocity" in info[0]:
            x_velocity = info[0]["x_velocity"]
            # Save the velocity to a file with the specified name in the directory
            with open(self.velocity_filepath, 'a') as f:
                f.write(f"{x_velocity}\n")
        return True


#    train(env, sb3_algo, reward_func, island_id, generation_id, counter)

def train(env, sb3_algo,reward_func, island_id, generation_id, counter, velocity_path, model_checkpoint_path, output_path, log_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_timesteps=0
    velocity_callback = VelocityLoggerCallback(
            velocity_filepath=velocity_path,  # Directly use the full file path
            verbose=1,
        )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_checkpoint_path, exist_ok=True)
    
    reward_callback = RewardLoggerCallback(log_dir, verbose=1)

    
    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == 'custom_net':
        model = SAC(CustomSACPolicy, env, verbose=1, device=device, tensorboard_log=log_dir)

    TIMESTEPS = 500
    #total_timesteps = 3000000
    total_timesteps = 1000 
    

    while current_timesteps < total_timesteps:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=[velocity_callback, reward_callback])
        current_timesteps += TIMESTEPS
        model_save_path = os.path.join(model_checkpoint_path, f"{sb3_algo}_{current_timesteps}.zip")
        model_save_path = os.path.join(
            model_checkpoint_path, 
            f"{sb3_algo}_{generation_id}_{counter}_{current_timesteps}.zip"
        )
        model.save(model_save_path)
        env.render()
        
        

        
    
# env=HumanoidEnv(
#         reward_fn_path=reward_fn_path,
#         counter=counter,
#         iteration=iteration,
#         group_id=group_id,
#         llm_model=llm_model,
#         baseline=baseline,
#         render_mode=None
#     )   

def run_training(reward_func, island_id, generation_id, counter, reward_history_file, model_checkpoint_file, fitness_file,velocity_file, output_path, log_dir):
    gymenv = HumanoidEnv(
        reward_func_str=reward_func,
        counter=counter,
        generation_id=generation_id,
        island_id=island_id,
        reward_history_file=reward_history_file,
        model_checkpoint_file=model_checkpoint_file,
        velocity_file=velocity_file,
    )
    sb3_algo='SAC'

    env = Monitor(gymenv)  # Ensure monitoring
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=100.0,clip_reward=100) # if norm_obs or norm_reward is true you need to save the state of the vecnormalize when loading the model weights.
    train(env, sb3_algo, reward_func, island_id, generation_id, counter, velocity_file, model_checkpoint_file, output_path, log_dir) 
    
    #return velocity_filepath
    
    

 