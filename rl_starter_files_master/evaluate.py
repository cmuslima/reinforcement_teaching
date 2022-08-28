import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
from rl_starter_files_master.student_utils import device, get_model_dir, Agent, make_env
from rl_starter_files_master.student_utils import device
import numpy as np



def evaluate_task(args, env_name):
    print('in eval')
    
    # Set device

    if args.debug:
        print(f"Device: {device}\n")

    # Load environments

    
    env = make_env(env_name, args.evaluation_seed + 10000)
    if args.debug:
        print("Environments loaded\n")

    # Load agent

    model_dir = get_model_dir(args.model_folder_path, args)
    try:
        agent = Agent(env.observation_space, env.action_space, model_dir,
                            argmax=args.argmax, num_envs=args.procs,
                            use_memory=args.memory, use_text=args.text)
    except:
        return 0, None
    #agent.return_weights()
    if args.debug:
        print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    #obs = env.reset()

    log_done_counter = 0
    log_episode_return = np.zeros(args.num_evaluation_episodes)
    #print('log_episode_return', log_episode_return)
    #log_episode_num_frames = torch.zeros(args.procs, device=device)
    all_actions = []
    all_values = []
    all_obs = []
    num_steps = 0
    while log_done_counter <args.num_evaluation_episodes:
        obs = env.reset()
        cum_reward = 0
        #print('log counter', log_done_counter)
        #env.render('human')
        while(True):
            #env.render('human')
            obs = [obs]
            action, value = agent.get_actions(obs)
            obs, reward, done, _ = env.step(action)

            #print('action', action)
            #print('value', value)
            #print('obs', obs)
            action = action.squeeze()
            value = value.squeeze()
            all_actions.append(action)
            all_values.append(value)
            all_obs.append(obs['image'])
            #print('obs', obs)
           
            #print(np.shape(obs))
            

            #log_episode_return += torch.tensor(reward, device=device, dtype=torch.float)
            #log_episode_num_frames += torch.ones(args.procs, device=device)

            cum_reward+=reward
            num_steps+=1
            if done:
                #print('done', cum_reward, 'num_steps', num_steps)
                log_episode_return[log_done_counter] = cum_reward
                log_done_counter += 1
                #print(log_episode_return)
                break
                #logs["return_per_episode"].append(log_episode_return.item())
                #logs["num_frames_per_episode"].append(log_episode_num_frames.item())

            #mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
            #log_episode_return *= mask
            #log_episode_num_frames *= mask

    env.close()
    params = agent.return_weights()
    end_time = time.time()

    #print('at of eval')
    #print('average return', np.mean(log_episode_return))
    #print('average num_steps', num_steps/args.num_evaluation_episodes)
    return np.mean(log_episode_return), params #all_actions, all_values, all_obs