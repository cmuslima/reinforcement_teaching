
import sys
import argparse
import utils
from run_training_loop import run_train_loop
from run_eval_loop import run_evaluation_loop



if __name__ == '__main__':
    print('\n\n\n\n\n\n\n main called \n\n\n\n\n\n\n\n')

    parser = argparse.ArgumentParser()

    parser.add_argument('--rootdir', type=str) 
    parser.add_argument('--exp_type', type=str, default = 'curriculum') #this should always remain the same
    parser.add_argument('--setting', type=str, default = 'RL') #this should always remain the same
    parser.add_argument('--env', type=str, default= 'four_rooms') #**this changes depending on the student task of interest
    parser.add_argument('--max_time_step', type=int) 
    parser.add_argument('--SR', type=str)
    parser.add_argument('--reward_function', type=str)
    parser.add_argument('--alpha', type=float, default= 1) #not needed
    #parser.add_argument('--teacher_evaluation_seed', type=int, default= 30)
    #parser.add_argument('--student_evaluation_seed', type=int, default= 0)

    parser.add_argument('--student_transfer', type=int, default= 0)
    parser.add_argument('--student_lr_transfer', type=int, default= 0)
    parser.add_argument('--student_NN_transfer', type=int, default= 0)
    parser.add_argument('--random_student_seed', type=int, default= 1)
    parser.add_argument('--student_discount', type=float, default= .99)
    parser.add_argument('--student_eps', type=float, default= .01)


    parser.add_argument('--teacher_agent', type=str, default= 'DQN')
    parser.add_argument('--teacher_eps_start', type=float, default= .5)
    parser.add_argument('--teacher_eps_decay', type=float, default= .99)
    parser.add_argument('--teacher_eps_end', type=float, default= .01)

    parser.add_argument('--teacher_buffersize', type=int)
    parser.add_argument('--teacher_batchsize', type=int)
    parser.add_argument('--teacher_lr', type=float, default= .001)
    parser.add_argument('--teacher_episodes', type=int) 

    #teacher network args

    parser.add_argument('--student_input_size', type=int)
    parser.add_argument('--student_output_size', type=int)
    parser.add_argument('--PE_hidden_size_input', type=int) 
    parser.add_argument('--PE_hidden_size_output', type=int) 
    parser.add_argument('--teacher_network_hidden_size', type=int) 
    parser.add_argument('--three_layer_network', type=bool) 


    parser.add_argument('--one_hot_action_vector', type=int, default = 1)#bool value
    parser.add_argument('--easy_initialization', type=int, default = 1)#bool value
    parser.add_argument('--reward_log', type=bool) #not used
    parser.add_argument('--normalize', type=bool) #not used


    parser.add_argument('--student_episodes', type=int)
    parser.add_argument('--num_training_episodes', type=int)
    parser.add_argument('--num_evaluation_episodes', type=int)
    parser.add_argument('--student_type', type=str)
    parser.add_argument('--two_buffer', type=int, default = 0) #not used #bool value
    parser.add_argument('--multi_controller', type=int, default = 0)#not used #bool value
    parser.add_argument('--multi_controller_v2', type=int, default = 0)#not used #bool value
    parser.add_argument('--clear_buffer', type=int, default = 0)#not used #bool value

    parser.add_argument('--multi_students', type=int, default = 0) #bool value
    parser.add_argument('--tabular', type=int, default = 0) #bool value
    parser.add_argument('--goal_conditioned', type=int, default = 0)#not used #bool value

    parser.add_argument('--stagnation_threshold', type=int, default= 3)#not used
    parser.add_argument('--LP_threshold', type=float, default= .05)#not used
    #parser.add_argument('--percent_change', type=bool, default = False)

    #types of teachers during evaluation:
    parser.add_argument('--random_curriculum', type=int, default = 0)#bool value
    parser.add_argument('--target_task_only', type=int, default = 0)#bool value
    parser.add_argument('--trained_teacher', type=int, default = 0)#bool value
    parser.add_argument('--handcrafted', type=int, default = 0)#bool value


    parser.add_argument('--debug', type=int, default= 0)#bool value
    parser.add_argument('--num_runs_start', type=int, default= 0)
    parser.add_argument('--num_runs_end', type=int, default= 10)
    parser.add_argument('--num_runs', type=int, default= 10)

    parser.add_argument('--num_student_processes', type=int, default= 1) #this is only used for fetch reach +open ai baselines
    parser.add_argument('--MP', type=int, default = 0)#bool value


    parser.add_argument('--training', type=int, default = 0) #bool value
    parser.add_argument('--evaluation', type=int, default = 0)#bool value
    parser.add_argument('--plotting', type=int, default = 0)#bool value
    parser.add_argument('--average', type=int, default = 0)#bool value
    parser.add_argument('--saving_method', type=str, default = 'exceeds_average')
    parser.add_argument('--folder_name', type=str, default = 'None')
    parser.add_argument('--hyper_param_sweep', type=int, default = 0)#bool value





    #Training params for PPO student
    parser.add_argument("--algo", type = str, default= 'ppo',
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    # parser.add_argument("--model", default='FourRoomsCL', 
    #                     help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=1,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")
    parser.add_argument("--updates", type=int, default=25,
                        help="number of frames of training (default: 1e7)")#was 50
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--student_batch_size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=128,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--student_lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    parser.add_argument("--evaluation_seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="action with highest probability is selected")


    # open AI baseline students
    parser.add_argument('--alg', help='Algorithm', type=str, default='her')
    parser.add_argument('--total_timesteps', type=float, default=10000), #100050
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=2, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--target_task', type=str, default='FetchPush-v1')


    #plotting arguments
    parser.add_argument('--single_baseline_comp', type=int, default=1)#bool value
    parser.add_argument('--comparing_scores', type=int, default=1)#bool value
    parser.add_argument('--plot_best_data', type=int, default=1)#bool value
    parser.add_argument('--stagnation', type=int, default=1)#bool value
    parser.add_argument('--AUC', type=int, default = 1)#bool value #AUC is area under curve, used for plotting



    args = parser.parse_args()

    if args.env == 'maze': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.rows = 11
        args.columns = 16
        args.student_num_actions = 4
        args.max_time_step = 39
        args.num_evaluation_episodes = 30
        args.num_training_episodes = 10
        args.student_lr = .5
        args.student_input_size = 2
        args.student_output_size = 4
        args.PE_hidden_size_input = 32
        args.PE_hidden_size_output = 32
        args.teacher_network_hidden_size = 64
        args.tabular = True
        args.teacher_episodes = 300
        args.student_episodes = 100 #*150
        args.three_layer_network = False
        args.stagnation = True #not used
        args.num_runs = 30
        args.student_type = 'q_learning'
    elif args.env == 'cliff_world': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.rows = 4
        args.columns = 12
        args.student_num_actions = 4
        args.max_time_step = 1000
        args.num_evaluation_episodes = 30
        args.num_training_episodes = 5
        args.student_lr = .5
        args.student_eps = .1
        args.student_input_size = 2
        args.student_output_size = 4
        args.PE_hidden_size_input = 32
        args.PE_hidden_size_output = 32
        args.teacher_network_hidden_size = 64
        args.tabular = True
        args.teacher_episodes = 400
        args.student_episodes = 150
        args.three_layer_network = False
        args.normalize = True
        print('args.env', args.env)
        args.student_type = 'q_learning'
    elif args.env == 'four_rooms':
        args.num_runs = 30
        args.tabular = False
        args.student_input_size = 243
        args.student_output_size = 3
        args.PE_hidden_size_input = 64
        args.PE_hidden_size_output = 64
        args.teacher_network_hidden_size = 128
        args.student_num_actions = 3
        args.max_time_step = 40
        args.num_evaluation_episodes = 40
        args.num_training_episodes = 25
        args.teacher_episodes = 100
        args.student_episodes = 50
        args.student_type = 'PPO'
        args.saving_method = 'exceeds_average'
        args.memory = args.recurrence > 1
        args.target_task = f'MiniGrid-Simple-4rooms-0-v0'
        #utils.make_dir(args, f'{args.rootdir}/models')
        args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}64_128'
        args.three_layer_network = False
        

    
    if args.env == 'fetch_reach_3D_outer': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.num_evaluation_episodes = 80
        args.student_input_size = 10
        args.student_output_size = 4
        args.PE_hidden_size_input = 64
        args.PE_hidden_size_output = 64
        args.teacher_network_hidden_size = 128
        args.teacher_episodes = 50
        args.student_episodes = 50 #change this back
        args.saving_method = 'exceeds_average'
        args.student_type = 'DDPG'
        args.three_layer_network = False
        args.tabular = False
        args.num_env = 1
        args.num_runs = 30
    if args.env == 'fetch_push':
        args.num_evaluation_episodes = 80
        args.student_input_size = 25
        args.student_output_size = 4
        args.PE_hidden_size_input = 64
        args.PE_hidden_size_output = 64
        args.teacher_network_hidden_size = 128
        args.teacher_episodes = 75
        args.student_episodes = 50
        args.saving_method = 'exceeds_average'
        args.student_type = 'DDPG'
        args.three_layer_network = False
        args.num_env = 2

    

    if args.hyper_param_sweep:
        learning_rates = [.001,.005]
        buffer_sizes = [100]
        batch_sizes = [128,256] 
        teacher_state_rep = 'L2T' #choices are buffer_policy (our method), buffer_q_table (our method), L2T, params
        teacher_rf_list = ['L2T'] # simple_LP (our method), LP, target_task_score, 0_target_task_score, cost, L2T'
        
        #L2T state + L2T reward = Fan et al (2018) method
        #params state + cost reward = Narvekear (2017) method
        if args.evaluation:
            args.teacher_episodes = 1
            args.teacher_eps_start = 0
            if args.random_curriculum == True or args.handcrafted == True or args.target_task_only:
                args.teacher_agent = 'None'
        for rf in teacher_rf_list:
            for buffer in buffer_sizes:
                for batch_size in batch_sizes:
                    for lr in learning_rates:
                        
                        args.SR = teacher_state_rep
                        args.reward_function = rf
                        args.teacher_batchsize = batch_size
                        args.teacher_lr = lr
                        args.teacher_buffersize = buffer
                        
                        #this is really only used for the PPO student
                        args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}_target'

                        args.rootdir = utils.get_rootdir(args, args.SR)
                        utils.make_dir(args, args.rootdir)
                        print(f'Root dir = {args.rootdir}')
                        if args.training:
                            run_train_loop(args)
                        if args.evaluation:
                            run_evaluation_loop(args)
    else:
        if args.evaluation:
            args.teacher_episodes = 1
            args.teacher_eps_start = 0
        args.rootdir = utils.get_rootdir(args, args.SR)
        utils.make_dir(args, args.rootdir)

        print('args.evaluation', args.evaluation)
        if args.training:
            run_train_loop(args)
        if args.evaluation:
            run_evaluation_loop(args)

                        
    
 


    # if args.plotting:
    #     learning_rates = [ .001]
    #     buffer_sizes = [100]
    #     batch_sizes = [128] 
    #     SR = ['random'] #'L2T' #'buffer_policy', 'buffer_q_table', 'params' need to do these with L2T
    #     rf = ['random'] 
    #     student_lrs = [.0001, .001, .01, .25]
    #     for state_rep in SR:
    #         # if 'buffer' not in state_rep:
    #         #     buffer_sizes = [1] #this is because we don't need to do any loops over buffer size for the not behavior embedded state reps
    #         for reward in rf:
    #             for lr in learning_rates:
    #                 for buffer in buffer_sizes:
    #                     for batch in batch_sizes:     
    #                         #for student_lr in student_lrs:
    #                         #args.student_lr = student_lr  
    #                         args.teacher_batchsize = batch
    #                         args.teacher_lr = lr
    #                         args.teacher_buffersize = buffer
    #                         args.SR = state_rep
    #                         args.reward_function = reward
    #                         args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}'
                            
    #                         args.rootdir = utils.get_rootdir(args, args.SR)
    #                         print('args.rootdir on config', args.rootdir)
    #                         #average.average_data(args)

    #     #plotting_graphs_updated.determine_normal_dis(args)
    #     #plotting_graphs_updated.plot_actions(args)
    #     #average.quick_plot(args)
    #     #plotting_graphs_updated.plot_single_baseline(args)
    #     #teacher_data_plot.plot_single_baseline(args)
    #     #plotting_graphs_updated.p_testing_area_under_curve(args)
    #     #average.average_data(args)
    #     # print(area_under_curve)
    #     #plotting_graphs_updated.t_testing(args)


