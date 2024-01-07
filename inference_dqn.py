import gym
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from DQN import ActorNetwork
from gym.wrappers import Monitor

ENV_ID = 'CartPole-v0'

def inference(model_path, video_path):
    env = gym.make(ENV_ID)
    env = Monitor(env, 
                  video_path, 
                  force=True)

    model = ActorNetwork(env.observation_space.shape[0], 32, env.action_space.n, torch.nn.functional.relu)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_reward = 0
    state = env.reset()
    done = False

    while not done:
        state_var = torch.tensor(np.array([state]), dtype=torch.float32)
        action = torch.argmax(model(state_var)).item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    print(f"Total Reward: {total_reward}")

    env.close()
    
    return total_reward

if __name__ == "__main__":
    model_path = 'output/2023-12-24_15-32-05/model/CartPole-v0_dqn_20000_2023-12-24_15-32-05.pth'
    
    VERSION = model_path.split('/')[1]
    OUT_PATH = Path(r'./output') / VERSION
    VIDEO_PATH = OUT_PATH / 'video'
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    VIDEO_PATH.mkdir(parents=True, exist_ok=True)

    video_dir = VIDEO_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    inference(model_path, video_dir)
