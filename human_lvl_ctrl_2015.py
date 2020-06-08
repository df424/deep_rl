
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %% Fix the keras issue.
import tensorflow as tf
config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction=0.9
tf.keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# %%
import gym
import matplotlib.pyplot as plt

from deep_rl.agent import RLAgent
from deep_rl.preprocessing import DQNAtariPreprocessor
from deep_rl.models import AtariDQNQNet
from deep_rl.agent.value_functions import DeepQNetwork

# %% Create the environment
env = gym.make('SpaceInvaders-v0')

# %% Setup the agent.

agent = RLAgent(
    observation_preprocessor=DQNAtariPreprocessor(),
    action_value_function=DeepQNetwork(AtariDQNQNet(env.action_space.n))
)

# %%
observation = env.reset()
reward = 0

for i in range(1000):
    env.render()
    action = agent.update(observation=observation, reward=reward)
    observation, reward, done, info = env.step(action=action)

env.close()

# %%
