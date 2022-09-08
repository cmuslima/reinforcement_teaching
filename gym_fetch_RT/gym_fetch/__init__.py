from gym.envs.registration import register

# Standard Environments
for reward_type in ['sparse']:
    if reward_type == 'dense':
        suffix = 'Dense'
    elif reward_type == 'sparse':
        suffix = 'Sparse'
    else:
        suffix = 'VerySparse'

    #for i, terminate_condition in enumerate([[False, False, .01], [True, False, .05], [False, True, .1], [True, True, .15]]):
    for i, terminate_condition in enumerate([[False, False, .01], [False, False, .04], [False, False, .08], [False, False, .12],[False, False, .15],[False, False, .2], [False, False, .24], [False, False, .28], [False, False, .32], [False, False, .36]]):
    #for i, terminate_condition in enumerate([[False, False, .15], [False, False, .03], [False, False, .06], [False, False, .09],[False, False, .12],[False, False, .15], [False, False, .24], [False, False, .28], [False, False, .32], [False, False, .36]]):

        kwargs = {
            'two_dim': True,
            'outer': False,
            'reward_type': reward_type,
            'terminate_success': terminate_condition[0],
            'terminate_fail': terminate_condition[1],
            'target_offset': terminate_condition[2]
            }


        register(
            id='FetchReach{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch_RT.gym_fetch.envs:FetchReachEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )


    for i, terminate_condition in enumerate([[False, False, .01], [False, False, .03], [False, False, .06], [False, False, .09],[False, False, .12],[False, False, .15], [False, False, .18], [False, False, .21], [False, False, .24]]): # 
        #print('in here')
        kwargs = {
            'two_dim': True,
            'outer': True,
            'reward_type': reward_type,
            'terminate_success': terminate_condition[0],
            'terminate_fail': terminate_condition[1],
            'target_offset': terminate_condition[2]
            }
    
       
        register(
            id='FetchReachOuter{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch_RT.gym_fetch.envs:FetchReachEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )
        #print('FetchPush{}-v{}'.format(suffix, i + 2))

        
        #print(id)
        # register(
        #     id='FetchPush{}-v{}'.format(suffix, i + 2),
        #     entry_point='gym_fetch.envs:FetchPushEnv',
        #     kwargs=kwargs,
        #     max_episode_steps=50,
        # )

        # register(
        #     id='FetchHook{}-v{}'.format(suffix, i + 2),
        #     entry_point='gym_fetch.envs:FetchHookEnv',
        #     kwargs=kwargs,
        #     max_episode_steps=100,
        # )


    for i, terminate_condition in enumerate([[False, False, .04], [False, False, .06], [False, False, .08], [False, False, .10],[False, False, .12], [False, False, .15], [False, False, .18], [False, False, .21], [False, False, .24]]): # 
        
        kwargs = {
            'two_dim': False,
            'outer': False,
            'reward_type': reward_type,
            'terminate_success': terminate_condition[0],
            'terminate_fail': terminate_condition[1],
            'target_offset': terminate_condition[2]
            }


        register(
            id='FetchReach3D{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch_RT.gym_fetch.envs:FetchReachEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )

        kwargs = {
            'two_dim': False,
            'outer': True,
            'reward_type': reward_type,
            'terminate_success': terminate_condition[0],
            'terminate_fail': terminate_condition[1],
            'target_offset': terminate_condition[2]
            }
    
       
        register(
            id='FetchReachOuter3D{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch_RT.gym_fetch.envs:FetchReachEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )
        #print('registering fetch reach 3d')
        #print(i, terminate_condition)