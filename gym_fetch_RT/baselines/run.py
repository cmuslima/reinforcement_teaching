import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
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

def bulid_target_task(args):
    env = build_env(args)
    return env


def train(args, extra_args, policy):
    print('in train from run.py')
    assert args.evaluation_only == False
    print(f'all args = {args}')
    #print(f'extra args = {extra_args}')
    env_type, env_id = get_env_type(args)
    print(f'env_type = {env_type}, env_id = {env_id}')
    #print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    #print(f'total time steps = {total_timesteps}')
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    #print('network', args.network)
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    #print(f'all alg kwargs = {alg_kwargs}')

    print('about to start the learn function')
    model, episode_data, params = learn(policy= policy,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
    #print('episode data')
    #states = episode_data[0]
    #print(episode_data[0].keys())
    #print(episode_data[0]['o'])
    #print(np.shape(episode_data[0]['o']))
    #print(episode_data[0]['u'])
    #print(np.shape(episode_data[0]['u']))
    #s = np.squeeze(episode_data[0]['o'])
    #print(np.shape(s))
    #print(episode_data[0]['o'])
    #print(s[0])
    extract_data(episode_data)
    args.evaluation_only = True
    assert args.evaluation_only == True
    target_task_env = bulid_target_task(args)
    return model, env, target_task_env

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
        #print('states i collected')
        #print(states)
       # print('real states')
        #print(episode_data[0]['o']) 
    assert len(states) == len(states)
    return states, actions

def build_env(args):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

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
        #print(f'building env')
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    if args.evaluation_only:
        env_id = args.target_task
        print('inside the target task only', env_id)
    else:
        env_id = args.env
        env_type = args.env

    #if args.env_type is not None:
        #return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

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
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    print('alg_module', alg_module)
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


def main(args):
    success_rate_during_training = []
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    no_curr = True
    random_curr = False
    if no_curr:
        print('no curriculum')
        env_lists = ['FetchReachSparse-v7']*1
    if no_curr == False:
        print(f'should be using handcrafted curriculum')
        v2_list = ['FetchReachSparse-v2']*5
        v3_list = ['FetchReachSparse-v3']*5
        v4_list = ['FetchReachSparse-v4']*9
        v5_list = ['FetchReachSparse-v5']*9
        v6_list = ['FetchReachSparse-v6']*10
        v7_list = ['FetchReachSparse-v7']*12
        
        env_lists =v2_list+v3_list+v4_list+v5_list+v6_list+v7_list
        assert len(env_lists) == 50
        #env_lists = ['FetchReachSparse-v2','FetchReachSparse-v3', 'FetchReachSparse-v4', 'FetchReachSparse-v5', 'FetchReachSparse-v6']
    if random_curr:
        print('using a random curriculum')
        randon_numbers = random.choices(range(2, 11), k = 50)
        env_lists = []
        for re in randon_numbers:
            env_name = f'FetchReachSparse-v{re}'
            env_lists.append(env_name)


    model = None
    for env_iter in range(0,len(env_lists)):
        #arg_parser = common_arg_parser()
        print(f'Training Iteration # {env_iter}')
        args.evaluation_only = False
        args.env = env_lists[env_iter]
        print('env being used', args.env)
        #print('args.evaluation_only', args.evaluation_only)
        extra_args = parse_cmdline_kwargs(unknown_args)

        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            rank = 0
            configure_logger(args.log_path)
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            configure_logger(args.log_path, format_strs=[])

        model, env, target_task_env = train(args, extra_args, model)

        if args.save_path is not None and rank == 0:
            save_path = osp.expanduser(args.save_path)
            model.save(save_path)

        if args.play:
            # args.evaluation_only = T
            logger.log("Running trained model on the target task")
            obs = target_task_env.reset()

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
                #env.render()
                done_any = done.any() if isinstance(done, np.ndarray) else done
                if done_any:
                    
                    for i in np.nonzero(done)[0]:
                        print('i', i)
                        print('episode_rew={}'.format(episode_rew[i]))
                        episode_rew_list.append(episode_rew[i])
                        #print(f'Number of failtures = {episode_rew_list.count(-50)/count}')
                        count+=1

                        episode_rew[i] = 0
                    if count > 25:
                        print(f'Final Number of failtures = {episode_rew_list.count(-50)/(count-1)}')
                        success_rate_during_training.append(episode_rew_list.count(-50)/(count-1))
                        break
            
            env.close()
            
            
    
    print('success rate throughout training', success_rate_during_training)
    return model

if __name__ == '__main__':
    main(sys.argv)
