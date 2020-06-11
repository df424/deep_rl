
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %%
import gym
import os

from deep_rl.agent import DQNAtariAgent
from deep_rl.experiment import Experiment
from deep_rl.agent.dqn_agent import DQNAtariQNet_Nature

# %% Create the environment
env = gym.make('Breakout-v0')

# %% Setup the agent.
agent = DQNAtariAgent(
    env.action_space.n, 
    discount_rate=0.99,
    replay_buffer_size=100000,
    replay_buffer_warmup=100000,
    action_value_function=DQNAtariQNet_Nature(env.action_space.n),
    optimizer_params = {'lr':0.00025, 'momentum':0.95, 'eps':0.01}
)

# %%
experiment = Experiment(agent=agent, env=env, output_dir='./output/experiment2')
experiment.run(50000000)