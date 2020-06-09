
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
import tqdm.autonotebook as tqdm

# %% Create the environment
env = gym.make('SpaceInvaders-v0')

# %% Setup the agent.
agent = DQNAtariAgent(env.action_space.n, discount_rate=0.99)

# %%
observation = env.reset()
action = 0
done = False
reward_sum = 0
max_reward = 0
all_rewards = {}

pbar = tqdm.trange(30000000, desc=f'reward=0, max_reward=0, plays=0', leave=True)

for i in pbar:
    if done:
        # Store the maximu reward we achieved.
        max_reward = max(max_reward, reward_sum)
        # Store the reward we recieved.
        all_rewards[i] = reward_sum
        # Set the reward to 0 for the next run.
        reward_sum = 0
        # Reset the environment.
        env.reset()

        # Update the progress bar.
        pbar.set_description(f'reward={all_rewards[i]}, max_reward={max_reward}, plays={len(all_rewards)}')

    observation, reward, done, info = env.step(action=action)
    #env.render()
    # Only update our agent every 3 (according to the dqn paper.)
    if i % 3 == 0:
        action = agent.step(observation=observation, reward=reward, done=done)

    # Keep track of the reward.
    reward_sum += reward

    if i > 256*3 and i % 100 == 0:
        agent.update(256)

    if i % 10000*3 == 0:
        agent.save(f'chkpt-{i}.pt')

env.close()

# %%
