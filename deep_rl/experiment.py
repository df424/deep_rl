
# Needed for experiment...
from torch.utils.tensorboard import SummaryWriter

import random
import tqdm.autonotebook as tqdm
import gym
import numpy as np
import os

from deep_rl.agent import RLAgent


class Experiment():
    def __init__(self, 
        agent: RLAgent, 
        env: gym.Env,
        output_dir: str,
        max_no_ops: int = 30,
        batch_size: int = 32,
        eval_freq: int = 100000
    ):
        # Make sure the directorys we need exist.
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
            os.mkdir(os.path.join(output_dir, 'checkpoints'))

        # %% Create the tensorboard logger.
        self._writer = SummaryWriter(os.path.join(output_dir, 'logs'))
        self._output_dir = output_dir
        self._agent = agent
        self._env = env
        self._max_no_ops = max_no_ops
        self._batch_size = batch_size
        self._eval_freq = eval_freq
        self._best_performance = -1e10

    def run(self, n_frames: int):
        # experiment parameters.
        episode_count = 0

        # episode parameters
        done = False
        action = 0
        reward = 0
        best_reward = 0
        last_observation = observation = self._env.reset()
        episode_step = 0
        reward_sum = 0
        no_ops = random.randint(0, self._max_no_ops)
        loss_sum = 0
        num_updates = 0

        # Create a progress bar to loop over the frames.
        pbar = tqdm.trange(n_frames)
        for i in pbar:
            if i % self._eval_freq == 0:
                self._agent.save(os.path.join(self._output_dir, 'checkpoints', f'dqn_{i}.pt'))

            if done:
                # Write episode metrics.
                best_reward = max(best_reward, reward_sum) 
                avg_loss = loss_sum/num_updates

                # Log to the tensorboard.
                self._writer.add_scalar('Reward/train', reward_sum, episode_count)
                self._writer.add_scalar('BestReward/train', best_reward, episode_count)
                self._writer.add_scalar('AverageLoss/train', avg_loss, episode_count)

                episode_count += 1

                pbar.set_description(f'Episode={episode_count}, Reward={reward_sum}, BestReward={best_reward}, AvgLoss={avg_loss:.4f}')

                # reset episode parameters
                done = False
                action = 0
                reward = 0
                last_observation = observation = self._env.reset()
                episode_step = 0
                reward_sum = 0
                no_ops = random.randint(1, self._max_no_ops)
                loss_sum = 0
                num_updates = 0

            
            # Make it variable what frame we start on.
            observation, step_reward, done, info = self._env.step(action=action)

            # Merge the last two observations to avoid problems with atari flickering.
            no_flicker_observation = np.maximum(observation, last_observation)

            # Clip the reward to -1/1
            reward += max(-1, min(1, step_reward))

            # If we are not in the no-op period of the episode.
            if episode_step > no_ops and episode_step > 0:
                # only let the agent see every fourth frame.
                if i % 4 == 0:
                    action = self._agent.step(no_flicker_observation, reward, done)
                    reward_sum += reward
                    reward = 0
                    # Given the agent the operturnity to log some stuff.
                    self._agent.log_metrics(self._writer, i)

                # Only let the agent update every fourth frame that it sees.
                if i % 16 == 0:
                    loss = self._agent.update(self._batch_size)
                    self._writer.add_scalar('Loss/train', loss, i)
                    loss_sum += loss
                    num_updates += 1
            
            # Update episode variables
            last_observation = observation
            episode_step += 1