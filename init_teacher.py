import student_env_setup
from evaluation import train_evaluate_protocol
import init_student
import utils
import torch
import numpy as np


from exp3_bandit_teacher import exp3_bandit

from teacher_env import get_student_env_name
# 
import random
class initialize_teacher_functions():
    def __init__(self, args):
      
        self.teacher_total_steps = 0
     
    #you only build the teacher once at the start of the teacher training
    def build_teacher(self, args, seed, env_state_size, env_action_size, file=None, student_seed = None):
    
        #Here I am instantiating the teacher's learning algorithm 
        if args.teacher_agent == 'DQN':
        
            if 'buffer' in args.SR:
          
                from buffer_teacher_agent import DQNAgent      
                print('env_state_size', env_state_size, 'env_action_size', env_action_size)
                teacher_agent = DQNAgent(state_size=env_state_size, action_size = env_action_size, seed=seed, args= args) 
                
                if args.trained_teacher:
                    print("Uploading trained teacher model")
                    teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
            else:
           
                from teacher_agent import DQNAgent
                teacher_agent = DQNAgent(state_size=env_state_size, action_size = env_action_size, seed=seed, args = args) 

                if args.trained_teacher:
                    print("Uploading trained teacher model")
                    teacher_agent.qnetwork_local.load_state_dict(torch.load(file))

        
        elif args.teacher_agent == 'exp3_bandit':
            teacher_agent = exp3_bandit(env_action_size, args)

        elif args.teacher_agent == 'Random':
            teacher_agent = None
        elif args.teacher_agent == 'No_Teacher':
            teacher_agent = None
        return teacher_agent

    def get_teacher_first_action(self, args, teacher_action_size,teacher_action_list, target_task_index, target_task):
        if args.exp_type == 'curriculum':

        # easy_initialization sets the first task to an easy one, this does require prior knowledge. Otherwise you have to randomly assign the first action. 
            if args.teacher_agent == 'DQN':
                if args.env == 'maze':
                    task_index = 7
                elif args.env == 'cliff_world':
                    task_index = teacher_action_size-1
                elif args.env == 'four_rooms':
                    task_index = 5
                elif args.env == 'fetch_reach_3D_outer': #need to double check this 8/16
                    task_index = 0 
                elif args.env == 'fetch_push': 
                    task_index = 1 
    
            elif args.teacher_agent == 'Random':
                task_index = random.randint(0, teacher_action_size-1)

            elif args.teacher_agent == 'No_Teacher':
                task_index = target_task_index
                task_name = target_task

            task_name = get_student_env_name(task_index, teacher_action_list, args)

        
        return task_name,task_index




          

        

  
 




    def save_teacher_data(self, args, teacher_agent, return_list, teacher_return, seed):
        if args.training:
            dir = f'{args.rootdir}/teacher-data'
            utils.make_dir(args, dir)
            file_details = ['teacher_return_list', seed]
            model_name = utils.get_model_name(args, dir, file_details)
            utils.save_data(model_name,return_list)
            if args.saving_method == 'every_episode': #need to add s

                if args.teacher_agent == 'DQN':
                    dir = f'{args.rootdir}/teacher-data' 

                    if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
                        model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}.pth'
                    else:
                        model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}.pth'
                    
                    print(f'Saving {model_name}')
                    torch.save(teacher_agent.qnetwork_local.state_dict(), model_name)
                    
            elif args.saving_method == 'exceeds_average':
                
                N = 5
                dir = f'{args.rootdir}/teacher-data' 
                if len(return_list) > N and teacher_return > sum(return_list[(-1*N):])/N:
                    if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
                        model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}.pth'
                    else:
                        model_name = f'{dir}/teacher_agent_checkpoint_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{seed}.pth'
                    
                    print(f'Saving {model_name}')
                    torch.save(teacher_agent.qnetwork_local.state_dict(), model_name)
                


    

    def get_eps(self, args, eps):
        if args.training:
            self.teacher_total_steps+=1
            print('self.teacher_total_steps', self.teacher_total_steps)
            eps = eps
            if self.teacher_total_steps > args.teacher_batchsize:
                eps = max(args.teacher_eps_end, args.teacher_eps_decay*eps) 
        else:
            eps = 0
        return eps
    
    def update_teacher_score(self, args):
        if args.evaluation and env.student_success == False:
            self.teacher_score+=1
        return self.teacher_score
    
    def get_teacher_action(self, teacher_agent, args, env,  traj= None, eps= None):
        if args.exp_type == 'curriculum':
            if args.teacher_agent == 'DQN':
                
                if (args.evaluation and env.student_success):
                    task_index = env.target_task_index
                    task_name = env.target_task
                else:
                    print('selecting action from the DQN teacher')
                    task_index = teacher_agent.act(traj, args, eps)
                    print('task index', task_index)
                    task_name = get_student_env_name(task_index, env.teacher_action_list, args)

            elif args.teacher_agent == 'Random': #random teacher
                if (args.evaluation and env.student_success): 
                    task_index = env.target_task_index
                    task_name = env.target_task
                else:
                    task_index = random.choice(env.teacher_action_list)
                    task_name = get_student_env_name(task_index, env.teacher_action_list, args)


            elif args.teacher_agent == 'No_Teacher': #no teacher. student is learning the target task from stratch
                task_index = env.target_task_index
                task_name = env.target_task
                    
            elif args.teacher_agent == 'exp3_bandit':
                if args.exp_type == 'curriculum':
                    task_index = teacher_agent.act(args)
                    task_name = get_student_env_name(task_index, env.teacher_action_list, args)
                    if (args.evaluation and env.student_success):
                        print('using target tsk')
                        task_index = env.target_task_index
                        task_name = env.target_task
        return task_index, task_name

 


    
