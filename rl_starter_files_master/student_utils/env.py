import gym
import rl_starter_files_master.gym_minigrid
from rl_starter_files_master.gym_minigrid.wrappers import *
from rl_starter_files_master.gym_minigrid.envs.simple4rooms import SimpleFourRoomsEnv
# from gym_minigrid.envs.simplemaze import SimpleMazeEnv

def make_env(env_key, seed=None):
    env = gym.make(env_key)
   
    #env =  SymbolicObsWrapper(env)
    env = FullyObsWrapper(env) #FullyObsWrapper
    env.seed(seed)
    print('inside make env')
    print('env key', env_key)
    s = env.reset()
    
    print('state', s)
    # print(env.action_space)
    # print(env.observation_space)
    return env
