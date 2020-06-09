
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

# %% Create the environment
env = gym.make('SpaceInvaders-v0')

# %% Setup the agent.
agent = DQNAtariAgent(env.action_space.n, discount_rate=0.99)

# %%
observation = env.reset()
reward = 0
done = False

for i in range(100):
    env.render()
    action = agent.step(observation=observation, reward=reward, done=done)
    observation, reward, done, info = env.step(action=action)

    if i > 32:
        agent.update(32)

env.close()