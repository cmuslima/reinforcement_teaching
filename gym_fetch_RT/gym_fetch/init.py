from gym.envs.registration import register

# Standard Environments
def init_new_envs():
    for reward_type in ['dense', 'sparse', 'very_sparse']:
        if reward_type == 'dense':
            suffix = 'Dense'
        elif reward_type == 'sparse':
            suffix = 'Sparse'
        else:
            suffix = 'VerySparse'

        #for i, terminate_condition in enumerate([[False, False, .01], [True, False, .05], [False, True, .1], [True, True, .15]]):
        for i, terminate_condition in enumerate([[False, False, .01], [False, False, .04], [False, False, .08], [False, False, .12],[False, False, .16],[False, False, .2], [False, False, .24], [False, False, .28], [False, False, .32], [False, False, .36]]):

            kwargs = {
                'reward_type': reward_type,
                'terminate_success': terminate_condition[0],
                'terminate_fail': terminate_condition[1],
                'target_offset': terminate_condition[2]
                }

            # Fetch
            # register(
            #     id='FetchSlide{}-v{}'.format(suffix, i + 2),
            #     entry_point='gym_fetch.envs:FetchSlideEnv',
            #     kwargs=kwargs,
            #     max_episode_steps=50,
            # )
            
            # register(
            #     id='FetchPickAndPlace{}-v{}'.format(suffix, i + 2),
            #     entry_point='gym_fetch.envs:FetchPickAndPlaceEnv',
            #     kwargs=kwargs,
            #     max_episode_steps=50,
            # )

            register(
                id='FetchReach{}-v{}'.format(suffix, i + 2),
                entry_point='gym_fetch.envs:FetchReachEnv',
                kwargs=kwargs,
                max_episode_steps=50,
            )
            id='FetchReach{}-v{}'.format(suffix, i + 2)
            #print('id', id)
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
