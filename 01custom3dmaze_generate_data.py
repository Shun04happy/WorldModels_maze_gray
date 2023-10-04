# xvfb-run -s "-screen 0 1400x900x24" python 01_generate_data.py car_racing --total_episodes 4000 --time_steps 300

import numpy as np
import random
import config_custom3dmaze
#import matplotlib.pyplot as plt

from env import make_env

import argparse

DIR_NAME = './data/custom3dmaze_gray2/'
# DIR_NAME = './data/rollout1/'

def main(args):

    env_name = args.env_name
    total_episodes = args.total_episodes
    time_steps = args.time_steps
    render = args.render
    run_all_envs = args.run_all_envs
    action_refresh_rate = args.action_refresh_rate

    if run_all_envs:
        envs_to_generate = config_custom3dmaze.train_envs
    else:
        envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        env = make_env(current_env_name)  # <1>
        s = 0

        while s < total_episodes:

            episode_id = random.randint(0, 2**31 - 1)
            filename = DIR_NAME + str(episode_id) + ".npz"

            env.reset()

            env.render()

            observation = env.reset_obs()
            t = 0

            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []

            reward = 0
            done = False

            while t < time_steps:  # and not done:
                if t % action_refresh_rate == 0:
                    action = config_custom3dmaze.generate_data_action(t, env)  # <2>

                # observation = config_custom2dmaze.adjust_obs(observation)  # <3>

                # obs_sequence.append(observation.copy())
                # action_sequence.append(action)
                # reward_sequence.append(reward)
                # done_sequence.append(done)
                env.render()
                observation, reward, done, info = env.step(action)  # <4>
                
                observation = config_custom3dmaze.adjust_obs(observation)
                # observation = np.split(observation, 3, axis=1)
                obs_sequence.append(observation.copy())
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
                t = t + 1

                
                    
                if reward == 1:
                    env.reset()
                #     obs_sequence.append(observation.copy())
                #     action_sequence.append(action)
                #     reward_sequence.append(reward)
                #     done_sequence.append(done)
                #     break
                
                if render:
                    env.render()

            print("Episode {} finished after {} timesteps".format(s, t))

            np.savez_compressed(filename, obs=obs_sequence, action=action_sequence,
                                reward=reward_sequence, done=done_sequence)  # <4>

            s = s + 1

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('env_name', type=str, help='name of environment')
    parser.add_argument('--total_episodes', type=int, default=200,
                        help='total number of episodes to generate per worker')
    parser.add_argument('--time_steps', type=int, default=300,
                        help='how many timesteps at start of episode?')
    parser.add_argument('--render', default=0, type=int,
                        help='render the env as data is generated')
    parser.add_argument('--action_refresh_rate', default=20, type=int,
                        help='how often to change the random action, in frames')
    parser.add_argument('--run_all_envs', action='store_true',
                        help='if true, will ignore env_name and loop over all envs in train_envs variables in config_custom2dmaze.py')

    args = parser.parse_args()
    main(args)
