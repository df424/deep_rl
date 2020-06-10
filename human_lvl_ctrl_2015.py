
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %%
import gym
import matplotlib.pyplot as plt

from deep_rl.agent import DQNAtariAgent
from deep_rl.experiment import Experiment

# %% Create the environment
env = gym.make('Breakout-v0')

# %% Setup the agent.
agent = DQNAtariAgent(env.action_space.n, discount_rate=0.99)

# %%
experiment = Experiment(agent, env)
experiment.run(50000000)

# %%
import numpy as np

# moving avg
def mov_avg(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

# %%
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.arange(len(experiment.history['rewards'])), experiment.history['rewards'])
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Total Reward vs Episode')
fig.savefig('./results/exp2_reward.png')
# %%
fig = plt.figure()
plt.plot(np.arange(len(experiment.history['rewards'])), mov_avg(experiment.history['rewards'], 100))
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Total Reward vs Episode (Moving Window) n=100')
fig.savefig('./results/exp2_smoothed_reward.png')

# %%
