
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %%
import gym
import os
import torch

from deep_rl.agent import DQNAtariAgent
from deep_rl.experiment import Experiment

# %% Create the environment
env = gym.make('Breakout-v0')

# %% Setup the agent (Comment out the optimizer parameter to duplciate experiment 1).
agent = DQNAtariAgent(
    env.action_space.n, 
    discount_rate=0.99,
    optimizer_params = {'lr':0.00025, 'momentum':0.95, 'eps':0.01}
)

# %%
experiment = Experiment(agent=agent, env=env, output_dir='./output/experiment3')
experiment.run(50000000)