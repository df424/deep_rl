
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %%
import gym
import matplotlib.pyplot as plt
import os

from deep_rl.agent import DQNAtariAgent
from deep_rl.experiment import Experiment
from torch.utils.tensorboard import SummaryWriter

# %%
EXPERIMENT_NAME = 'Experiment1'

# %% Create the environment
env = gym.make('Breakout-v0')

# %% Create the tensorboard logger.
writer = SummaryWriter(os.path.join('./output', EXPERIMENT_NAME))

# %% Setup the agent.
agent = DQNAtariAgent(env.action_space.n, discount_rate=0.99)

# %%
experiment = Experiment(writer=writer, agent=agent, env=env)
experiment.run(50000000)