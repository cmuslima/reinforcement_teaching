
#requirements:
#gym, numpy, scipy, tensorflow, click, mpi4py

import os
from gym.wrappers import FlattenObservation
import click
import numpy as np
import json
import tensorflow as tf
from mpi4py import MPI
from common.tf_util import get_session
import logger
from common import set_global_seeds, tf_util
from common.mpi_moments import mpi_moments
import her.experiment.config as config
from her.rollout import RolloutWorker
from her.her import learn
import gym
from gym_fetch.envs import FetchPushEnv
from common.cmd_util import make_vec_env
from bench import Monitor
logger_dir = logger.get_dir()
print('finished importing')
mpi_rank, subrank = 0, 0
#env = gym.make('FetchReachSparse-v4')
# obs = env.reset()
# done = False

# def policy(observation, desired_goal):
#     # Here you would implement your smarter policy. In this case,
#     # we just sample random actions.
#     return env.action_space.sample()

# for i in range(0,10):
#     #print(i)
#     obs = env.reset()
#     done = False
#     #print('obs', obs)
#     while not done:
#         action = policy(obs['observation'], obs['desired_goal'])
#         obs, reward, done, info = env.step(action)
#         env.render()

#alg_kwargs['network'] = 'mlp'
kwargs = {}
config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
get_session(config=config)
env_id = 'FetchReachSparse-v4'
env_type = 'FetchReachSparse-v4'
seed = 0
env = make_vec_env(env_id, env_type, 1, seed)
#env = gym.make('FetchReachSparse-v4')
#env = FlattenObservation(env)
#env = Monitor(env,
                #logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                #allow_early_resets=True)
learn('mlp', env, 1000)

# @click.command()
# @click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
# @click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
# @click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
# @click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
# @click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='none', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
# @click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
# @click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
# def main(**kwargs):
#     learn(**kwargs)


# if __name__ == '__main__':
#     main()
