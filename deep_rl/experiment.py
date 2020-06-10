
# Needed for experiment...
import random
import tqdm.autonotebook as tqdm
import gym
import numpy as np

from deep_rl.agent import RLAgent

class Experiment():
    def __init__(self, 
        agent: RLAgent, 
        env: gym.Env,
        max_no_ops: int = 30,
        batch_size: int = 32,
    ):
        self._agent = agent
        self._env = env
        self._max_no_ops = max_no_ops
        self._batch_size = batch_size
        self.history = {
            'rewards': [],
            'best_reward': 0
        }

    def run(self, n_frames: int):
        # experiment parameters.
        epside_count = 1

        # episode parameters
        done = False
        action = 0
        reward = 0
        last_observation = observation = self._env.reset()
        episode_step = 0
        reward_sum = 0
        no_ops = random.randint(0, self._max_no_ops)

        # Create a progress bar to loop over the frames.
        pbar = tqdm.trange(n_frames)
        for i in pbar:
            if done:
                # Store episode metrics.
                self.history['rewards'].append(reward_sum)
                self.history['best_reward'] = max(self.history['best_reward'], reward_sum) 
                epside_count += 1

                pbar.set_description(f'Reward={reward_sum}, BestReward={self.history["best_reward"]}')

                # reset episode parameters
                done = False
                action = 0
                reward = 0
                last_observation = observation = self._env.reset()
                episode_step = 0
                reward_sum = 0
                no_ops = random.randint(0, self._max_no_ops)

            
            # Make it variable what frame we start on.
            observation, step_reward, done, info = self._env.step(action=action)

            # Merge the last two observations to avoid problems with atari flickering.
            no_flicker_observation = np.maximum(observation, last_observation)

            # Clip the reward to -1/1
            reward += max(-1, min(1, step_reward))

            # If we are not in the no-op period of the episode.
            if episode_step > no_ops:
                # only let the agent see every fourth frame.
                if i % 4 == 0:
                    action = self._agent.step(no_flicker_observation, reward, done)
                    reward_sum += reward
                    reward = 0

                # Only let the agent update every fourth frame that it sees.
                if i % 16 == 0:
                    self._agent.update(self._batch_size)
            
            # Update episode variables
            last_observation = observation
            episode_step += 1