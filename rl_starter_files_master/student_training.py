import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
from rl_starter_files_master.student_utils import device, get_model_dir, Agent, make_env, get_status, save_status
from rl_starter_files_master.student_utils.other import synthesize
from rl_starter_files_master.student_utils.format import get_obss_preprocessor
from rl_starter_files_master.model import ACModel
import numpy as np
from rl_starter_files_master.gym_minigrid.wrappers import *



# Set run dir
def student_train(args, env_name):
    
    if args.debug:
        print(f'Start of student')
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model_folder_path or default_model_name
    #print('model_name', model_name)
    model_dir = get_model_dir(model_name, args)

    #print(f"Model Dir = {model_dir}")
    # Load loggers and Tensorboard writer

    #txt_logger = student_utils.get_txt_logger(model_dir)
    #csv_file, csv_logger = student_utils.get_csv_logger(model_dir)
    #tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    #txt_logger.info("{}\n".format(" ".join(sys.argv)))
    #txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    #student_utils.seed(args.seed)

    # Set device

    #txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(make_env(env_name, args.seed + 10000 * i))
    #txt_logger.info("Environments loaded\n")

    
    # Load training status

    try:
        status = get_status(model_dir)
        #print('status', status)
    except OSError:

        status = {"num_frames": 0, "update": 0}
  
    #txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    #txt_logger.info("Observations preprocessor loaded")

    # Load model
   
    acmodel = ACModel(obs_space, envs[0].action_space, args.memory, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    #txt_logger.info("Model loaded\n")
    #txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.student_lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.student_lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.student_batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    #txt_logger.info("Optimizer loaded\n")

    # Train model
    #print('Beginning to train model')
    num_frames = status["num_frames"]
    update = status["update"]
   
    start_time = time.time()

    tot_updates = 0

    while tot_updates < args.num_training_episodes:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
       
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1
        tot_updates+=1
        
        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = synthesize(logs["return_per_episode"])
            rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            # txt_logger.info(
            #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            #     .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            # if status["num_frames"] == 0:
            #     csv_logger.writerow(header)
            # csv_logger.writerow(data)
            # csv_file.flush()

            # for field, value in zip(header, data):
            #     tb_writer.add_scalar(field, value, num_frames)

        
        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            if args.debug:
                print("Saving model")
                print(f'Model dir {model_dir}')
            status = {"num_frames": num_frames, "update": update,
                    "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            save_status(status, model_dir)
            #txt_logger.info("Status saved")
    envs[0].close()
    #print(f"End of training")
    # print('exps.obs', list(exps.obs.keys()))
    # #a = list(exps.obs['text'])
    #print(exps.value)
    exps_copy = dict(exps)
    #print(list(exps_copy.keys()))
    obs = list(exps_copy['obs'].values())
    values = exps.value
    actions = exps.action
    #print('values', np.shape(values))
    #print('actions', np.shape(actions), actions)
    obs = obs[0]
    shape = np.shape(obs)
    #print(shape)
    obs = np.reshape(obs, (shape[0],243))
    #print(np.shape(b))
 
    
    return rreturn_per_episode['mean'], obs, values, actions
    