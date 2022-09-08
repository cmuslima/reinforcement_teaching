import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
#sys.path.insert(1, './baselines')
from gym_fetch_RT.baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv

from gym_fetch_RT.baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from gym_fetch_RT.baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from gym_fetch_RT.baselines.common.tf_util import get_session
from gym_fetch_RT.baselines import logger
from importlib import import_module
from gym_fetch_RT.gym_fetch import init
from gym_fetch_RT.baselines.her.rollout import RolloutWorker
from gym_fetch_RT.baselines.her.her import mpi_average
#init.init_new_envs()


#arg = args()

import random

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    #print('env', env)
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)
    #print(env, env_type)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


#all args = Namespace(alg='her', env='FetchPush-v1', env_type=None, gamestate=None, log_path=None, network=None, num_env=None, num_timesteps=2500.0, play=True, reward_scale=1.0, save_path=None, save_video_interval=0, save_video_length=200, seed=None)


def init_student(args):
    #print('in the init student function')
    #utils.set_global_seeds(seed, args)
    # np.random.seed(0)
    # tf.set_random_seed(0)
    # random.seed(0)
    env_dict = build_all_envs(args)
    print(env_dict)
    print('inside init student')
    return env_dict

def train(args, policy, env_id, env_type, env_dict, training_complete, config_params, dims):
    print('env dict', env_dict)
    #print('in train from run.py')
    #assert args.evaluation_only == False
    #print(f'all args = {args}')
    #print(f'extra args = {extra_args}')
    #print('config params', config_params)
    #print('dims', dims)
    evaluation_only = False # False initally because we are just training now. 
    env_type, env_id = get_env_type(args, env_id, env_type)
    #print('after get_env_type function')
    print(f'env_type = {env_type}, env_id = {env_id}')
   
    total_timesteps = int(args.total_timesteps)
    #print(f'total time steps = {total_timesteps}')
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    #alg_kwargs.update(extra_args)
    #print('got the learn function')
    #print('before env = env_dict[env_id]')
    env = env_dict[env_id] #build_env(args, env_id, env_type)
    #env = build_env(args, env_id, env_type)
    print('env', env)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    #print('network', args.network)
    #print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    #print(f'all alg kwargs = {alg_kwargs}')

    #print('about to start the learn function')

    model, episode_data, params, config_params, dims = learn(policy= policy,
        env=env,
        seed=seed,
        training_complete=training_complete,
        total_timesteps=total_timesteps, 
        params=config_params,
         dims=dims,
        **alg_kwargs
    )
    if training_complete:
        return None, None, None, None, None, None

    states, actions = extract_data(episode_data) 
    print(np.shape(states))
    #target_task_env = bulid_target_task(args)
    return model, states, actions, params, config_params, dims

def extract_data(episode_data):
    states = []
    actions = []
    for d in episode_data:
        s = np.squeeze(d['o'])
        a = np.squeeze(d['u'])
        for i in range(49): #max_times = 50
            #print(s[i])
            states.append(s[i])
            actions.append(a[i])
            #print(a[i])
        #print('states i collected')
        #print(states)
       # print('real states')
        #print(episode_data[0]['o']) 
    #print(actions)
    #print(len(actions))
    #print(np.shape(actions[0]))
    assert len(states) == len(actions)
    return states, actions

def build_env(args, env_id, env_type):
    #print('insisde build env')
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args, env_id, env_type)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        #print('inside the build_env function')
        #print(f'building env')
        # config = tf.ConfigProto(allow_soft_placement=True,
        #                        intra_op_parallelism_threads=1,
        #                        inter_op_parallelism_threads=1)
        # config.gpu_options.allow_growth = True
        # new_student = False
        # get_session(new_student = new_student, config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)
        #print('got the env', env)
        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env

def build_all_envs(args):
   # print("bulding all envs once")
    env_dict = {}

    if args.env == 'fetch_push':
        for i in range(1,10):
            
            env_name = f'FetchPush-v{i}'
            env_id = env_name
            env_type = env_name
            env = build_env(args, env_id, env_type)
            env_dict[env_name] = env
    print('env dict')
    print(env_dict)
    if args.env == 'fetch_reach_2D':
        for i in range(2,12):
            
            env_name = f'FetchReachSparse-v{i}'
            env_id = env_name
            env_type = env_name
            env = build_env(args, env_id, env_type)
            env_dict[env_name] = env
    if args.env == 'fetch_reach_2D_outer':
        
        for i in range(2,11):
            #print('env number', i)
            env_name = f'FetchReachOuterSparse-v{i}'
            env_id = env_name
            env_type = env_name
            env = build_env(args, env_id, env_type)
            env_dict[env_name] = env
        #print('building the target task')
        env_name = f'FetchReachSparse-v{6}'
        env_id = env_name
        env_type = env_name
        env = build_env(args, env_id, env_type)
        env_dict[env_name] = env
    if args.env == 'fetch_reach_3D':
        for i in range(2,12):
            
            env_name = f'FetchReach3DSparse-v{i}'
            env_id = env_name
            env_type = env_name
            env = build_env(args, env_id, env_type)
            env_dict[env_name] = env
    if args.env == 'fetch_reach_3D_outer':
        
        for i in range(2,11):
            
            env_name = f'FetchReachOuter3DSparse-v{i}'
            env_id = env_name
            env_type = env_name
            env = build_env(args, env_id, env_type)
            env_dict[env_name] = env
            print('env number', i, env_name)
        #print('building the target task')
        env_name = f'FetchReach3DSparse-v{6}'
        env_id = env_name
        env_type = env_name
        env = build_env(args, env_id, env_type)
        env_dict[env_name] = env
        
        # env_name = f'FetchReach-v{1}'
        # env_id = env_name
        # env_type = env_name
        # env = build_env(args, env_id, env_type)
        # env_dict[env_name] = env    
    #not sure if you need to create the session in the build env place
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                         intra_op_parallelism_threads=1,
    #                         inter_op_parallelism_threads=1)
    # config.gpu_options.allow_growth = True
    # new_student = False
    # get_session(True, config=config)
    #print('done building envs')
    return env_dict

    

def get_env_type(args, env_id, env_type):
    # if evaluation_only:
    #     env_id = args.target_task
    #     env_type = args.target_task
    #     print('inside the target task only', env_id)
    # else:
    env_id = env_id
    env_type = env_type

    #print('got env type', env_id, env_type)
    #if args.env_type is not None:
        #return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

        #print(env_type)
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
        #print('env_id in this loop', env_id)
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    print('returning', env_type, env_id)
    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['gym_fetch_RT.baselines', alg, submodule]))
        #print('alg_module', import_module)
    except ImportError:
        # then from rl_algs
        #print('alg', alg, 'submolde', submodule)
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    #print('alg_module', alg_module)
    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


# def main(model):
#     success_rate_during_training = []
#     # configure logger, disable logging in child MPI processes (with rank > 0)

#     args = common_arg_parser()
    
#     #args, unknown_args = arg_parser.parse_known_args(args)
#     no_curr = True
#     random_curr = False
#     if no_curr:
#         print('no curriculum')
#         env_lists = ['FetchReachSparse-v7']*1
#         print(env_lists)
#     if no_curr == False:
#         print(f'should be using handcrafted curriculum')
#         v2_list = ['FetchReachSparse-v2']*5
#         v3_list = ['FetchReachSparse-v3']*5
#         v4_list = ['FetchReachSparse-v4']*9
#         v5_list = ['FetchReachSparse-v5']*9
#         v6_list = ['FetchReachSparse-v6']*10
#         v7_list = ['FetchReachSparse-v7']*12
        
#         env_lists =v2_list+v3_list+v4_list+v5_list+v6_list+v7_list
#         assert len(env_lists) == 50
#         print(env_lists[0:10])
#         #env_lists = ['FetchReachSparse-v2','FetchReachSparse-v3', 'FetchReachSparse-v4', 'FetchReachSparse-v5', 'FetchReachSparse-v6']
#     if random_curr:
#         print('using a random curriculum')
#         randon_numbers = random.choices(range(2, 11), k = 50)
#         env_lists = []
#         for re in randon_numbers:
#             env_name = f'FetchReachSparse-v{re}'
#             env_lists.append(env_name)

#         print(env_lists[0:10])
    
    
#     #model = None
#     for env_iter in range(0,len(env_lists)):
#         print('ok')
#         #arg_parser = common_arg_parser()
#         print(f'Training Iteration # {env_iter}')
        
#     #     #args.evaluation_only = False
#     #     #args.env = env_lists[env_iter]
        
#         env_id = env_lists[env_iter]
#         env_type = env_lists[env_iter]
#         print('env being used', env_id)

#     #     #print('args.evaluation_only', args.evaluation_only)
#     #     #extra_args = parse_cmdline_kwargs(unknown_args)
#     #     #print('extra args', extra_args)
#         if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
#             rank = 0
#             configure_logger(args.log_path)
#         else:
#             rank = MPI.COMM_WORLD.Get_rank()
#             configure_logger(args.log_path, format_strs=[])

        
#         model,ddpg_states, ddpg_actions, params, target_task_env = train(args, model, env_id, env_type, training_complete)

#         if args.save_path is not None and rank == 0:
#             save_path = osp.expanduser(args.save_path)
#             model.save(save_path)

#         print('args.play', args.play)
       
#         evaluation_only = True
#         if evaluation_only:
#             # args.evaluation_only = T
#             logger.log("Running trained model on the target task")
#             obs = target_task_env.reset()

#             state = model.initial_state if hasattr(model, 'initial_state') else None
            
#             dones = np.zeros((1,))
#             episode_rew_list = []
#             episode_rew = np.zeros(target_task_env.num_envs) if isinstance(target_task_env, VecEnv) else np.zeros(1)
            
#             count = 1
#             while True:
#                 if state is not None:
#                     actions, _, state, _ = model.step(obs,S=state, M=dones)
#                 else:
#                     actions, _, _, _ = model.step(obs)

#                 obs, rew, done, _ = target_task_env.step(actions)
#                 episode_rew += rew
#                 #env.render()
#                 done_any = done.any() if isinstance(done, np.ndarray) else done
#                 if done_any:
                    
#                     for i in np.nonzero(done)[0]:
#                         print('i', i)
#                         print('episode_rew={}'.format(episode_rew[i]))
#                         episode_rew_list.append(episode_rew[i])
#                         #print(f'Number of failtures = {episode_rew_list.count(-50)/count}')
#                         count+=1

#                         episode_rew[i] = 0
#                     if count > args.num_evaluation_episodes:
#                         print(f'Final Number of failtures = {episode_rew_list.count(-50)/(count-1)}')
#                         success_rate_during_training.append(episode_rew_list.count(-50)/(count-1))
#                         break
            
#             target_task_env.close()
            
            
    
#     print('success rate throughout training', success_rate_during_training)
#     return model,ddpg_states, ddpg_actions, params



#need to make sure I'm seeding the student only once

def new_student_eval(eval_env, policy, params, dims):
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }
    n_test_rollouts = 80
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(eval_env, policy, dims, **eval_params)

    eval_episode_data = []
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        eval_episode = evaluator.generate_rollouts()
        eval_episode_data.append(eval_episode)

    eval_states, eval_actions = extract_data(eval_episode_data) 
    #print('eval_states', eval_states)
    #print('eval_actions', eval_actions)
    #print(np.shape(eval_actions))
    success_rate = mpi_average(evaluator.current_success_rate())
    #print(f'success rate on task = {success_rate}')
    return success_rate, eval_states, eval_actions
def student_training(model, env_name, args, env_dict,training_complete,config_params, dims):
    print('inside student training')
    #print(f'config_params {config_params}')
    #print(f'dims = {dims}')
    success_rate_during_training = []
    # configure logger, disable logging in child MPI processes (with rank > 0)

    
    
    #print('ok')
    #arg_parser = common_arg_parser()
    
    
#     #args.evaluation_only = False
#     #args.env = env_lists[env_iter]
    
    env_id = env_name
    env_type = env_name
    #print('env being used', env_id)


    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    #training_complete = False #I will have to call train one more time after the stuedent solves the target_task
    model,ddpg_states, ddpg_actions, params, config_params, dims = train(args, model, env_id, env_type, env_dict, training_complete, config_params, dims)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    return model,ddpg_states, ddpg_actions, params, config_params, dims

def student_evalution(args, model, env_name, env_dict, config_params, dims):
    
    # configure logger, disable logging in child MPI processes (with rank > 0)

    if model == None:
        return 0, None, None
    env_id = env_name
    env_type = env_name
    #print('env being used', env_id)

    env = env_dict[env_id]
    #env = build_env(args, env_id, env_type)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    
    


    print(f"Running trained model on the task {env_id}")
    # obs = env.reset()

    # state = model.initial_state if hasattr(model, 'initial_state') else None
    
    # dones = np.zeros((1,))
    # episode_rew_list = []
    # episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
    
    # count = 1
    # while True:
    #     if state is not None:
    #         actions, _, state, _ = model.step(obs,S=state, M=dones)
    #     else:
            
    #         actions, _, _, _ = model.step(obs)

    #     obs, rew, done, _ = env.step(actions)
    #     episode_rew += rew
    #     # if render:
    #     env.render()
    #     done_any = done.any() if isinstance(done, np.ndarray) else done
    #     if done_any:
            
    #         for i in np.nonzero(done)[0]:
    #             #print('i', i)
    #             #print('episode_rew={}'.format(episode_rew[i]))
    #             episode_rew_list.append(episode_rew[i])
    #             #print(f'Number of failtures = {episode_rew_list.count(-50)/count}')
    #             count+=1

    #             episode_rew[i] = 0
    #         if count > args.num_evaluation_episodes:
    #             #print(f'Percent  of failtures = {episode_rew_list.count(-50)/(count-1)}')

    #             break
    
    # env.close()
        
    # print('success rate throughout training', 1-episode_rew_list.count(-50)/(count-1))

    success_rate, eval_states, eval_actions = new_student_eval(env, model, config_params, dims)
    #print('success rate throughout training', 1-episode_rew_list.count(-50)/(count-1))
    print('success rate throughout training', success_rate)
    return success_rate, eval_states, eval_actions # , #1-episode_rew_list.count(-50)/(count-1)

def play(args, model, env_name, env_dict, config_params, dims):
    
    # configure logger, disable logging in child MPI processes (with rank > 0)

    if model == None:
        return 0, None, None
    env_id = env_name
    env_type = env_name
    #print('env being used', env_id)

    env = env_dict[env_id]
    #env = build_env(args, env_id, env_type)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    
    


    print(f"Running trained model on the task {env_id}")
    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    
    dones = np.zeros((1,))
    episode_rew_list = []
    episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
    
    count = 1
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        # if render:
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            
            for i in np.nonzero(done)[0]:
                #print('i', i)
                #print('episode_rew={}'.format(episode_rew[i]))
                episode_rew_list.append(episode_rew[i])
                #print(f'Number of failtures = {episode_rew_list.count(-50)/count}')
                count+=1

                episode_rew[i] = 0
            if count > 10:
                #print(f'Percent  of failtures = {episode_rew_list.count(-50)/(count-1)}')

                break
    
    env.close()

    return # , #1-episode_rew_list.count(-50)/(count-1)