import argparse
import time
from rl_starter_files_master.student_utils import device, get_model_dir, Agent, make_env, seed
from rl_starter_files_master.student_utils import device
import numpy as np
# Parse arguments



# Run the agent
def visualize(args):
    model = 'buffer_policy_128_0.001_simple_LP_100_0'
    model_dir = get_model_dir(model, args)
    print('model_dir', model_dir)
    task_index = 0
    env_name = f'MiniGrid-Simple-4rooms-{task_index}-v0'
    env = make_env(env_name, args.evaluation_seed + 10000)
    agent = Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, use_memory=args.memory, use_text=args.text)
   
    print("Agent loaded\n")
    #if args.gif:
    #from array2gif import write_gif
    frames = []

    # Create a window to view the environment
    env.render('human')

    for episode in range(1000):
        obs = env.reset()
        num_steps = 0
        while True:
            env.render('human')
            # if args.gif:
            #     frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_steps+=1
            if done or env.window.closed:
                print(f'num_steps {num_steps}')
                break

        if env.window.closed:
            break

    # if args.gif:
    #     print("Saving gif... ", end="")
    #     write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    #     print("Done.")
