# pytorch-madrl

This project includes PyTorch implementations of Deep Q-Network algorithm for single agent.

- [ ] DQN

It is written in a modular way to allow for sharing code between different algorithms. In specific, each algorithm is represented as a learning agent with a unified interface including the following components:
- [ ] interact: interact with the environment to collect experience. Taking one step forward and n steps forward are both supported (see `_take_one_step_` and `_take_n_steps`, respectively)
- [ ] train: train on a sample batch
- [ ] exploration_action: choose an action based on state with random noise added for exploration in training
- [ ] action: choose an action based on state for execution
- [ ] value: evaluate value for a state-action pair
- [ ] evaluation: evaluation the learned agent

# Requirements

- python 3.6+
- gym
- pytorch

# Usage

To train a model:

```
$ python run_dqn_mod.py
```

To inference a model:

```
$ python inference_dqn.py
```

## Results
It's extremely difficult to reproduce results for Reinforcement Learning algorithms. Due to different settings, e.g., random seed and hyper parameters etc, you might get different results compared with the followings.

### DQN

![CartPole-v0](output/2023-12-24_15-32-05/chart/CartPole-v0_dqn.png)

# Acknowledgments
This project gets inspirations from the following projects:
- [ ] Ilya Kostrikov's [pytorch-a2c-ppo-acktr](https://github.com/ChenglongChen/pytorch-a2c-ppo-acktr) (kfac optimizer is taken from here)
- [ ] OpenAI's [baselines](https://github.com/openai/baselines)

# License
MIT