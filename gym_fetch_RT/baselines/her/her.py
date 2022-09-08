import os

import click
import numpy as np
import json
from mpi4py import MPI

from gym_fetch_RT.baselines import logger
from gym_fetch_RT.baselines.common import set_global_seeds, tf_util
from gym_fetch_RT.baselines.common.mpi_moments import mpi_moments
import gym_fetch_RT.baselines.her.experiment.config as config
from gym_fetch_RT.baselines.her.rollout import RolloutWorker
using_logger = True
#seed = 0
#set_global_seeds(seed)
def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def evaluation_only(evaluator, n_test_rollouts):
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()
    success_rate = mpi_average(evaluator.current_success_rate())
    return success_rate

def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, training_complete, **kwargs):

    if training_complete:
        print('training is complete inside HER')
        policy.close()
        return None, None, None
    rank = MPI.COMM_WORLD.Get_rank()
    #print('rank', rank)
    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    if using_logger:
        logger.info("Training...")
    best_success_rate = -1

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    #print('n_batches', n_batches)
    #print('n_test_rollouts',n_test_rollouts)
    # evaluation_only = False
    # if evaluation_only:
    #     success_rate = evaluation_only(evaluator, n_test_rollouts)
    #if evaluation_only == False:
    n_epochs = 1
    all_episode_info = []
    all_eval_episode_info = []
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for i in range(n_cycles):
            #print('cycle', i)
            #print('index of cycles', i)
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            #print('stored episode')
            all_episode_info.append(episode)
            for current_batch in range(n_batches):
                #print('training now', j)
                _, _, params = policy.train(current_batch, n_batches)
                #print(len(params))
            policy.update_target_net()


        #back here at next stop
        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            eval_episode = evaluator.generate_rollouts()
            

        #training related info
        if using_logger:
            for key, val in rollout_worker.logs('train'):
                logger.record_tabular(key, mpi_average(val))
            for key, val in evaluator.logs('test'):
                logger.record_tabular(key, mpi_average(val))
            for key, val in policy.logs():
                logger.record_tabular(key, mpi_average(val))
            logger.record_tabular('epoch', epoch)

            if rank == 0:
                logger.dump_tabular()

        # save the policy if it's better than the previous ones
        
        success_rate = mpi_average(evaluator.current_success_rate())
        #print(f'Success rate = {success_rate}')
        if rank == 0 and success_rate >= best_success_rate and save_path:
            best_success_rate = success_rate
            if using_logger:
                logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            if using_logger:    
                logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        #make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy, all_episode_info, params

def get_all_params_ready(policy, network, env, training_complete, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='None',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
):
    if policy == None:
        override_params = override_params or {}
        if MPI is not None:
            rank = MPI.COMM_WORLD.Get_rank()
            num_cpu = MPI.COMM_WORLD.Get_size()

        #print(f'MPI = {MPI}, rank = {rank}, num_cpu = {num_cpu}')
        # Seed everything.
        #rank_seed = seed + 1000000 * rank if seed is not None else None
        # set_global_seeds(seed)

        # Prepare params.
        
        params = config.DEFAULT_PARAMS.copy()
        
        #print('printing scope')
        #print(params)
        env_name = env.spec.id
        params['env_name'] = env_name
        #print('env name in her file', env_name)
        params['replay_strategy'] = replay_strategy
        print('replay stra', params['replay_strategy'] )
        if env_name in config.DEFAULT_ENV_PARAMS:
            params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
        params.update(**override_params)  # makes it possible to override any parameter
    
        #with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
            #json.dump(params, f)
        #print('config.DEFAULT_PARAMS', config.DEFAULT_PARAMS)
        params = config.prepare_params(params)
        params['rollout_batch_size'] = env.num_envs

        if demo_file is not None:
            params['bc_loss'] = 1
        params.update(kwargs)

        if using_logger:
            config.log_params(params, logger=logger)

        if num_cpu == 1:
            if using_logger:
                logger.warn()
                logger.warn('*** Warning ***')
                logger.warn(
                    'You are running HER with just a single MPI worker. This will work, but the ' +
                    'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
                    'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
                    'are looking to reproduce those results, be aware of this. Please also refer to ' +
                    'https://github.com/openai/baselines/issues/314 for further details.')
                logger.warn('****************')
                logger.warn()

        dims = config.configure_dims(params)
    print('got dims and params')
    print('params', params, dims)
    return params, dims
    #print("POLICY", policy)
def learn(*, policy, network, env, training_complete, total_timesteps, params, dims,
    seed=None,
    eval_env=None,
    replay_strategy='None',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None, 
    **kwargs
):

    # override_params = override_params or {}
    # if MPI is not None:
    #     rank = MPI.COMM_WORLD.Get_rank()
    #     num_cpu = MPI.COMM_WORLD.Get_size()

    # #print(f'MPI = {MPI}, rank = {rank}, num_cpu = {num_cpu}')
    # # Seed everything.
    # #rank_seed = seed + 1000000 * rank if seed is not None else None
    # # set_global_seeds(seed)

    # # Prepare params.
    
    # params = config.DEFAULT_PARAMS.copy()
    
    # #print('printing scope')
    # #print(params)
    # env_name = env.spec.id
    # params['env_name'] = env_name
    # print('env name in her file', env_name)
    # params['replay_strategy'] = replay_strategy
    # if env_name in config.DEFAULT_ENV_PARAMS:
    #     params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    # params.update(**override_params)  # makes it possible to override any parameter
   
    # #with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
    #      #json.dump(params, f)
    # #print('config.DEFAULT_PARAMS', config.DEFAULT_PARAMS)
    # params = config.prepare_params(params)
    # params['rollout_batch_size'] = env.num_envs

    # if demo_file is not None:
    #     params['bc_loss'] = 1
    # params.update(kwargs)

    # if using_logger:
    #     config.log_params(params, logger=logger)

    # if num_cpu == 1:
    #     if using_logger:
    #         logger.warn()
    #         logger.warn('*** Warning ***')
    #         logger.warn(
    #             'You are running HER with just a single MPI worker. This will work, but the ' +
    #             'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
    #             'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
    #             'are looking to reproduce those results, be aware of this. Please also refer to ' +
    #             'https://github.com/openai/baselines/issues/314 for further details.')
    #         logger.warn('****************')
    #         logger.warn()

    # dims = config.configure_dims(params)
    #print("POLICY", policy)
    if policy == None:
        #print('policy', policy)
        #print('inside config dfpg')
        params, dims = get_all_params_ready(policy, network, env, training_complete, total_timesteps,params, dims,**kwargs)
        #print('got params and dims')
        policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    #print('after rollout and eval params')
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    rollout_worker = RolloutWorker(env, policy, dims, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    print('num time steps')
    print(100 // 6 // rollout_worker.T // 2)
    print(200 // 6 // rollout_worker.T // 2)
    print(500 // 6 // rollout_worker.T // 2)

    print('number of epochs', n_epochs)
    #print('about to do the her train')
    
    policy, all_episode_info, ddpg_params = train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, training_complete= training_complete)

    return policy, all_episode_info, ddpg_params, params, dims

#@click.command()
#@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
#@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
#@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
#@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
#@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='none', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
#@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
#@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
