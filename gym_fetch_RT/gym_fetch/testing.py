import numpy as np
import gym
from gym_fetch.envs import FetchPushEnv



env = gym.make('FetchReachSparse-v2')
obs = env.reset()
done = False

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()

for i in range(0,10):
    print(i)
    obs = env.reset()
    done = False
    print('obs', obs)
    while not done:
        action = policy(obs['observation'], obs['desired_goal'])
        obs, reward, done, info = env.step(action)
        env.render()
    
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    # substitute_goal = obs[‘achieved_goal’].copy()
    # substitute_reward = env.compute_reward(
    #     obs[‘achieved_goal’], substitute_goal, info)
    # print(‘reward is {}, substitute_reward is {}’.format(
    #     reward, substitute_reward))

# import numpy as np
# import gym


# env = gym.make('FetchReach-v1')

# # Simply wrap the goal-based environment using FlattenDictWrapper
# # and specify the keys that you would like to use.
# env = gym.wrappers.FlattenDictWrapper(
#     env, dict_keys=['observation', 'desired_goal'])

# # From now on, you can use the wrapper env as per usual:
# ob = env.reset()
# print(ob.shape)  # is now just an np.array


# import numpy as np
# import gym


# env = gym.make('FetchReach-v1')
# obs = env.reset()
# done = False

# def policy(observation, desired_goal):
#     # Here you would implement your smarter policy. In this case,
#     # we just sample random actions.
#     return env.action_space.sample()

# while not done:
#     action = policy(obs['observation'], obs['desired_goal'])
#     obs, reward, done, info = env.step(action)
#     env.render()
#     # If we want, we can substitute a goal here and re-compute
#     # the reward. For instance, we can just pretend that the desired
#     # goal was what we achieved all along.
#     # substitute_goal = obs[‘achieved_goal’].copy()
#     # substitute_reward = env.compute_reward(
#     #     obs[‘achieved_goal’], substitute_goal, info)
#     # print(‘reward is {}, substitute_reward is {}’.format(
#     #     reward, substitute_reward))
