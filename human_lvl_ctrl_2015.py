
# 
# David L. Flanagan
# June 2, 2020
# In this notebook I reimplement the 2015 DQN paper by Minh et all.
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# 

# %%
import gym
from deep_rl.agent import RLAgent
from deep_rl.preprocessing.image import ImagePrepPipeline, ImageValueScalePrepStage, ImageResizePrepStage, Rgb2GrayscalePrepStage, ImageCropStage
import matplotlib.pyplot as plt

# %% Setup the agent.
prep = ImagePrepPipeline()
prep.add_stage(ImageResizePrepStage(new_size=(110, 84)))
prep.add_stage(ImageCropStage(new_size=(84, 84), orient='center'))
prep.add_stage(Rgb2GrayscalePrepStage())
prep.add_stage(ImageValueScalePrepStage(256))

agent = RLAgent()

# %%
env = gym.make('SpaceInvaders-v0')
observation = env.reset()


for i in range(1000):
    env.render()
    action = agent.update(observation=observation, reward=0)
    observation, reward, done, info = env.step(action=action)

    if i == 10:
        img = prep.prep(observation)
        plt.imshow(img, cmap='gray')
        print(img.shape)
        break

env.close()



# %%
