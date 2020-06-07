
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %%
import gym
from deep_rl.agent import RLAgent

# %%
env = gym.make('SpaceInvaders-v0')
observation = env.reset()

agent = RLAgent()

for _ in range(1000):
    env.render()
    action = agent.update(observation=observation, reward=0)
    observation, reward, done, info = env.step(action=action)

env.close()

# %%
