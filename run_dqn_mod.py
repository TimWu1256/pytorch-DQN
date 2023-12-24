from DQN import DQN
from common.utils import agg_double_list

import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch

MAX_EPISODES = 50000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# max steps in each episode, prevent from running too long
MAX_STEPS = 10000  # None

MEMORY_CAPACITY = 10000
BATCH_SIZE = 100
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

DONE_PENALTY = -10.

RANDOM_SEED = 2023

VERSION = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

OUT_PATH = Path(r'./output/dqn') / VERSION
MODEL_PATH = OUT_PATH / 'model'
LOG_PATH = OUT_PATH / 'log'
CHART_PATH = OUT_PATH / 'chart'

ENV_ID = 'CartPole-v1'

def run():
    env = gym.make(ENV_ID)  # Specify the directory to save videos
    env.seed(RANDOM_SEED)
    env_eval = gym.make(ENV_ID)
    env_eval.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    if len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    dqn = DQN(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
              done_penalty=DONE_PENALTY, critic_loss=CRITIC_LOSS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN)
    
    episodes = []
    eval_rewards = []
    while True:
        dqn.interact()
        if dqn.n_episodes >= EPISODES_BEFORE_TRAIN:
            dqn.train()

        if dqn.episode_done:
            if ((dqn.n_episodes + 1) % EVAL_INTERVAL == 0):
                rewards, _ = dqn.evaluation(env_eval, EVAL_EPISODES)
                rewards_mu, rewards_std = agg_double_list(rewards)
                print(f"Episode {dqn.n_episodes + 1}, Average Reward {rewards_mu:.2f}")
                episodes.append(dqn.n_episodes + 1)
                eval_rewards.append(rewards_mu)
                
            if not (dqn.n_episodes < MAX_EPISODES):
                break

    env.close()  # Close the environment after training is done
    
    # Save the model
    torch.save(dqn.actor.state_dict(), MODEL_PATH / f"{ENV_ID}_dqn_{MAX_EPISODES}_{VERSION}.pth")

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)
    np.savetxt(LOG_PATH / f"{ENV_ID}_dqn_episodes.txt", episodes)
    np.savetxt(LOG_PATH / f"{ENV_ID}_dqn_eval_rewards.txt", eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.title(f"{ENV_ID}")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DQN"])
    plt.savefig(CHART_PATH / f"{ENV_ID}_dqn.png")


if __name__ == "__main__":
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    CHART_PATH.mkdir(parents=True, exist_ok=True)
    
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
