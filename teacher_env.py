import student_env_setup
from evaluation import train_evaluate_protocol
import utils
import torch
import numpy as np
import math
import random
from init_student import initalize_single_student


def get_student_env_name(task_index, teacher_action_list, args):
    print(teacher_action_list, task_index)
    if args.exp_type == 'curriculum':
        if args.env == 'maze' or args.env == 'cliff_world':
            task_name = teacher_action_list[task_index]
        elif args.env == 'four_rooms':
            task_name = f'MiniGrid-Simple-4rooms-{task_index}-v0'
        elif args.env == 'fetch_reach_3D_outer' :
            task_name = f'FetchReachOuter3DSparse-v{task_index+2}' 
        elif args.env == 'fetch_push':
            task_name = f'FetchPush-v{task_index+1}' 
    #print('task name', task_name, task_index, teacher_action_list)
    return task_name

class teacher_environment():
    def __init__(self, args):
        
        self.LP_dict = dict()
        self.student_returns_dict = dict()
        self.student_success = False
        #teacher decay information
        self.target_task_success_threshold = None
        self.student_model = None
        self.student_config_params = None
        self.student_dims = None 
        if args.env == 'fetch_push':
            self.run, self.build_env = utils.import_modules(args)
        if args.env == 'fetch_reach_3D_outer':
            self.RT_run  = utils.import_modules(args)
        if args.env == 'four_rooms':
            self.student_train, self.evaluate_task = utils.import_modules(args)


    #This function is only used for the fetch envs + DDPG student algo by stable baselines. Their implemention using TF and the TF graph must be ''closed'' before you can stop training on a student and reset a new student. 
    def close_student(self, args):
        if args.student_type == 'DDPG':
            if args.env == 'fetch_push':
                self.run.student_training(self.student_model, self.student_config_params, self.student_dims, True, self.student_env_dict, self.target_task, args)
            else:
            
                self.RT_run.student_training(self.student_model, self.target_task, args, self.student_env_dict, True, self.student_config_params, self.student_dims)
        else:
            return


    #maybe done
    def build_teacher_env(self, args, seed):
        self.train_evaluate_protocol = train_evaluate_protocol(args)
        self.eval_flag = False
        self.set_teacher_action_list(args)
        self.initalize_teacher_state_space(args)
        
        self.teacher_total_steps = 0 
        
        
        self.build_student_env_dicts(args) #this is only required for the fetch reach + DDPG implementation. This just helps create all the fetch reach envs
        self.determine_target_task_success_threshold(args)

                ##need to figure this part out
        utils.set_global_seeds(seed, args)
    

    #this is only done once
    def build_student_env_dicts(self, args):

        self.student_env_dict = None
        if args.student_type == 'DDPG' and args.env != 'fetch_push':
            self.student_env_dict = self.RT_run.init_student(args) 
        if args.student_type == 'DDPG' and args.env == 'fetch_push':
            self.student_env_dict = self.build_env.main(args) 

        

#this occurs at every teacher episode
    def reset_env_data(self, args):
       
        if args.env == 'four_rooms' and args.student_type == 'PPO':
            utils.remove_dir(args, f'{args.rootdir}/storage/{args.model_folder_path}') #need to create a folder path
            print('removing directory')
    
        self.initalize_all_dicts()
        self.student_state_buffer = []
        self.student_params = 0
        self.student_type = args.student_type 
        self.student_model = None
    
        self.average_target_task_score = 0
        self.average_target_task_LP = 0
        print('Finished resetting the env dicts')

    
    def step(self, task_index, task_name, current_student_episode, args):
        reward = None
        traj_prime = None
        
        self.first_occurence_task_check(task_index, task_name, args)  #  # loop N times            

        source_task_score, target_task_score, source_task_training_score = self.student_train_protocol(task_name, args) # loop 1 times, with N processes
    
        LP = self.get_LP(source_task_score, task_index, args) #loop N times 

        self.update_teacher_dicts(task_index, source_task_score, target_task_score, LP, current_student_episode, args) #loop N times 

        if args.teacher_agent == 'DQN': 
            reward = self.get_teacher_reward(task_index, LP, target_task_score,source_task_training_score, current_student_episode, args) ##loop N times 
            traj_prime = self.get_traj_prime(task_index, source_task_score, target_task_score, current_student_episode, args)
                              
        done = self.find_termination(target_task_score, current_student_episode, args.student_episodes, args)
        
        if done:
            self.close_student(args)
    
        return traj_prime, reward, done, target_task_score
    def reset(self, args, teacher_agent_functions):
        self.student_agent = initalize_single_student(args)
        self.reset_env_data(args)


        student_id = None

        task_name, task_index = teacher_agent_functions.get_teacher_first_action(args, self.teacher_action_size,self.teacher_action_list, self.target_task_index, self.target_task)
        print('just finished getting my first teacher action')

        self.first_occurence_task_check(task_index, task_name, args)
        assert self.student_model == None and self.student_params == 0
        traj_prime, reward, done, target_task_score = self.step(task_index, task_name, 0, args)

        return task_index, traj_prime

   #maybe done yet
    def first_occurence_task_check(self, task_index, task_name, args):
        if task_index not in list(self.student_returns_dict.keys()):
            print(f'task index = {task_index} not in returns dict')
            average_score, _ = self.train_evaluate_protocol.evaluate(self.student_agent, task_name, args, self.student_model, self.student_env_dict, self.student_config_params, self.student_dims) 
            update_entry = {task_index: average_score}
            self.student_returns_dict.update(update_entry)

        

        

    def update_teacher_dicts(self, task_index, source_task_score, target_task_score, LP, current_student_episode, args):

        if args.normalize:
            source_task_score = utils.normalize(-(args.max_time_step+1), self.get_max_value(task_index), source_task_score)
            
        #print(f'Updating teacher dicts')
        update_entry = {task_index: source_task_score}
        #print(f'update entry = {update_entry}')
        

        self.student_returns_dict.update(update_entry)
        #print(self.student_returns_dict)
        self.update_LP_dict(task_index, LP)

        if current_student_episode == 0:
            self.average_target_task_score = 0
        else:
            self.average_target_task_score+= (1/current_student_episode)*(target_task_score-self.average_target_task_score)

        self.average_target_task_LP = target_task_score - self.average_target_task_score
        #print(self.LP_dict)


    
    def student_train_protocol(self, task_name, args):
        student_id = None
        #self.student_model, self.student_params this is only for ddpg student

        source_task_training_score, obss, q_values, actions, self.student_model, self.student_params, self.student_config_params, self.student_dims = self.train_evaluate_protocol.train(self.student_agent, self.student_type, task_name, args, self.student_model, self.student_env_dict, self.student_config_params, self.student_dims)
        print('finished training')
        if args.teacher_agent == 'DQN':
            if 'buffer' in args.SR:
                self.add_obs_to_buffer(obss, q_values, actions, args)
       
        #this part is only here b/c it was easier getting the student parameters in the PPO student's eval function compared to at the end of the PPO's training loop
        if args.student_type == 'DDPG':
            source_task_score, target_task_score, _ = self.train_evaluate_protocol.source_target_evaluation(student_id, self.student_agent, task_name, self.target_task, args, self.student_model, self.student_env_dict, self.student_config_params, self.student_dims)
        else:
            source_task_score, target_task_score, self.student_params= self.train_evaluate_protocol.source_target_evaluation(student_id, self.student_agent, task_name, self.target_task, args, self.student_model, self.student_env_dict, self.student_config_params, self.student_dims)
        
        print('finishing evaluting on target and source task ')
        return source_task_score, target_task_score, source_task_training_score




    #done
    def set_teacher_action_list(self, args):
    #we make the assumption that the first task in the teacher's action list is the target task
        self.teacher_action_list = []
        if args.env == 'maze':
            self.teacher_action_list = [np.array([10,4]),np.array([1,1]), np.array([5,1]), np.array([9,1]),np.array([7,5]), np.array([3,6]), np.array([5,10]), np.array([2,12]), np.array([10,8]),  np.array([10,14]),  np.array([7,13])]
            print('teacher_action_list', self.teacher_action_list)

        elif args.env == 'cliff_world':
            self.teacher_action_list = [np.array([3,0]),np.array([0,0]), np.array([0,3]), np.array([0,6]),np.array([0,9]),np.array([0,11]),np.array([2,2]),np.array([2,5]),np.array([2,8]), np.array([2,11])] #,np.array([1,2]), np.array([1,5]), np.array([1,9]) ]
        elif args.env == 'four_rooms':
            for i in range(0, 10):
                self.teacher_action_list.append(i)
        elif args.env == 'fetch_reach_3D_outer':
            for i in range(0, 9):
                self.teacher_action_list.append(i)
        elif args.env == 'fetch_push':
            for i in range(0,9):
                self.teacher_action_list.append(i)    

        self.teacher_action_size = len(self.teacher_action_list)
        print(f'Set teacher action list {self.teacher_action_list} with action size {self.teacher_action_size}')
        # if args.increase_decrease:
        #     self.teacher_action_size = 3
        self.get_target_task(args)


    #done
    def get_target_task(self, args):
        self.target_task = get_student_env_name(0, self.teacher_action_list,args)
        self.target_task_index = 0
        if args.env == 'fetch_reach_3D_outer': #need to check on this one 8/16
            self.target_task_index = 4
            self.target_task= 'FetchReach3DSparse-v6'
        if args.env == 'fetch_push':
            self.target_task = get_student_env_name(0, self.teacher_action_list,args)
            self.target_task_index = 0
        print(f'Set student target task {self.target_task} {self.target_task_index}')


    #should be done
    def initalize_teacher_state_space(self, args):

        if args.SR == 'action_return':
            if args.one_hot_action_vector:
                self.teacher_state_size =  1 + self.teacher_action_size #return + one hot encoding of the action
            else:
                self.teacher_state_size = 2 #action, return

        elif args.SR == 'L2T':
            print('teacher_action_size', self.teacher_action_size)
            self.teacher_state_size = 3 + self.teacher_action_size
            print()
        elif args.SR == 'loss_mismatch':
            print('teacher_action_size', self.teacher_action_size)
            self.teacher_state_size = 5 + self.teacher_action_size
           
    
        elif args.SR == 'params':
            if args.tabular:
                self.teacher_state_size = args.rows*args.columns*args.student_num_actions
            else:
    
                if args.student_type == 'PPO':
                    self.teacher_state_size = 43700

                elif args.student_type == 'DDPG' and args.env == 'fetch_reach_3D_outer':
                    self.teacher_state_size = 545290

                        
                elif args.student_type == 'DDPG' and args.env == 'fetch_push':
                    self.teacher_state_size = 560650 
      
        elif 'buffer' in args.SR:

            if args.tabular:
                student_env_state_size = 2 #(x,y) position

            else:
                if args.student_type == 'PPO':
                    student_env_state_size = 243

                elif args.student_type == 'DDPG' and args.env == 'fetch_reach_3D_outer':
                    student_env_state_size = 10 #wait this should be 10???

                        
                elif args.student_type == 'DDPG' and args.env == 'fetch_push':
                    student_env_state_size = 25

            #student_env_state_size is the input size and args.student_num_actions is the output size
            #when the student's env is continous action space student_num_actions is simply the size of the student action vector
            self.teacher_state_size = args.teacher_buffersize*(student_env_state_size+args.student_num_actions)
   
   
    #The purpose of this is to keep incrementing the curriculum length until the stuent solves the target task. Once the student solves the target task, the teacher will just select the target task only. 
    #env stuff
    def evaluation_true(self, args, teacher_episode):
        if teacher_episode == args.teacher_episodes:
            self.eval_flag = True

   
    
    def get_traj_prime(self, task_index, source_task_score, target_task_score, student_episode, args):
        

        if args.SR == 'action_return':
            if args.one_hot_action_vector:
                one_hot_vector = self.get_one_hot_action_vector(task_index, 1, self.teacher_action_size)
                if args.debug:
                    print(f'one hot vector for action {task_index} = {one_hot_vector}')
                traj_prime = one_hot_vector + [source_task_score]
            else:
                traj_prime = [task_index, source_task_score] 

            traj_prime = np.array(traj_prime)
        
        elif args.SR == 'L2T':
        # We collect several simple features, such as passed mini-batch number (i.e., iteration), the average historical training loss and historical validation accuracy. They are
        # proven to be effective enough to represent the status of current student mode
            print('task index', task_index)
            print('teacher action size', self.teacher_action_size)
            one_hot_vector = utils.get_one_hot_action_vector(task_index, 1,  0, self.teacher_action_size)

            traj_prime = one_hot_vector + [source_task_score, target_task_score, student_episode] 
            traj_prime = np.array(traj_prime)
            print(traj_prime)
            traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
        elif args.SR == 'loss_mismatch':
        # We collect several simple features, such as passed mini-batch number (i.e., iteration), the average historical training loss and historical validation accuracy. They are
        # proven to be effective enough to represent the status of current student mode
            print('task index', task_index)
            print('teacher action size', self.teacher_action_size)

            one_hot_vector = utils.get_one_hot_action_vector(task_index, 1,  0, self.teacher_action_size)

            traj_prime = one_hot_vector + [source_task_score, target_task_score, self.average_target_task_score, self.average_target_task_LP, student_episode/args.student_episodes] 
            traj_prime = np.array(traj_prime)
            print(traj_prime)
            traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
        elif args.SR == 'params':
            if args.tabular:
                q_values = list(self.student_agent.q_matrix.values())
                traj_prime = np.array(q_values)
                traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
            else:
                traj_prime = np.array(self.student_params)
        elif args.SR == 'params_action':
            if args.tabular:
                q_values = list(self.student_agent.q_matrix.values())
                traj_prime = np.array(q_values)
                traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size))
            else:
                traj_prime = np.array(self.student_params)
            
        elif args.SR == 'params_student_type':
            if args.tabular:
                #print('in get trah')
                q_values = list(self.student_agent.q_matrix.values()) 
                traj_prime = np.array(q_values)
                #print('teacher_state_size', self.teacher_state_size)
                traj_prime = np.reshape(traj_prime, (1,self.teacher_state_size-2))
                if self.student_type == 'q_learning':
                    index = 0
                else:
                    index = 1
                one_hot_student_type = utils.get_one_hot_action_vector(index, 1, 0, 2)
                #print('one_hot_student_type', one_hot_student_type)
                student_type = np.reshape(one_hot_student_type, (1,2))
                
                traj_prime = np.concatenate((traj_prime, student_type), axis = 1)
                #print('self.student_type', self.student_type)
                #print('traj prime', traj_prime)
            else:
                traj_prime = np.array(self.student_params)
    
    
        elif 'buffer' in args.SR: #this buffer q table will now just look at the q value of the action chosen, not all q values 
            mini_state = []
            vals = []
            for i in range(0, args.teacher_buffersize):

                #randomly selecting a state from the buffer
                s = random.randint(0, len(self.student_state_buffer)-1)
                state = self.student_state_buffer[s]
                policy = self.policy_table[state] #policy or q-value
    
                value = np.concatenate((state, policy))
                #print(f'index = {s} state = {state}')
                mini_state.append(value)

            mini_state = np.array(mini_state)

            traj_prime = mini_state
        #print('traj prime', traj_prime)
        return traj_prime


        

    #this should occur at every teacher episode
    def initalize_all_dicts(self):
        for action in range(0, len(self.teacher_action_list)): 
            emptylist = list()
            self.LP_dict.update({action:emptylist})   
        self.student_returns_dict = dict()
        self.policy_table= dict()
        
    
    def update_LP_dict(self, task_index, LP_value):
        self.LP_dict[task_index].append(LP_value)


    #this should only occur once
    def determine_target_task_success_threshold(self, args):

        if args.env == 'four_rooms':
            self.target_task_success_threshold = .6
        if args.env == 'maze':
            self.target_task_success_threshold = (.99)**35
        if args.env == 'cliff_world':
            self.target_task_success_threshold = (-1)*20
        if args.env == 'fetch_reach_3D_outer':
            self.target_task_success_threshold = .9
        if args.env == 'fetch_push':
            self.target_task_success_threshold = .85
        if args.debug:
            print(f'self.target_task_success_threshold is {self.target_task_success_threshold}')
    
    

            
    def check_for_success_on_target_task(self, target_task_score, args):
        if target_task_score >= self.target_task_success_threshold:
            target_task_reward = 0
            if args.debug:
                print('agent success on target task')
            success = True
        else:
            if args.debug:
                print('agent failure at target task')
            target_task_reward = -1
            success = False
        
        return target_task_reward, success

   

    #Not done yet
    def get_teacher_reward(self, task_index, LP, target_task_score, source_task_training_score, student_sucesss_episode, args):
        target_task_reward, success = self.check_for_success_on_target_task(target_task_score, args)

        #SF = self.get_SF(task_index, args)

        #print('USING REWARD FUNCTION', args.reward_function)
        
        if args.reward_function == 'binary_LP':
            if LP > 0:
                reward = 1
            else:
                reward = -1
        
            print('LP', LP, 'reward', reward)
            
        if args.reward_function == 'target_task_score':
            reward = target_task_score
        if args.reward_function == '0_target_task_score':
            if success:
                reward = target_task_score
            else:
                reward = 0

        
        if args.reward_function == 'L2T':
            if success:
                reward = -math.log(student_sucesss_episode/args.student_episodes)
            else:
                reward = 0
            #print('reward', reward)
        if args.reward_function == 'Narvekar2018' or args.reward_function == 'Narvekar2017': 
            reward = -(source_task_training_score)

        if args.reward_function == "cost":
            if args.debug:
                print('using regular reward function')
            reward = target_task_reward # this will be 0 if the student solved the target task and -1 otherwise 

        elif args.reward_function == "LP":
            reward = args.alpha*LP 
            
        elif args.reward_function == 'simple_LP' or args.reward_function == 'simple_ALP':
            reward = -1 + args.alpha*LP 
        
        elif args.reward_function == 'LP_SF_log':
              
            if SF >= args.stagnation_threshold: #this says I have converged and I have still failed on the target task, so I should probably avoid this task
                reward = (-1)*math.log(10+SF) #target_task_reward acts as the -1 cost signal 
            else:
                reward = -1 + args.alpha*LP 
        elif args.reward_function == 'LP_log':

            reward = (-1)*math.log(10+SF) + args.alpha*LP 
        elif args.reward_function == 'LP_cost_relu':
            reward = target_task_reward - self.relu(SF) + args.alpha*LP 

        return reward

    def find_termination(self, target_task_score, current_episode, max_episode, args):
        #print(target_task_score, self.target_task_success_threshold)
        return_value = False
        print('max_episode', max_episode, 'current episode', current_episode)
        if current_episode == max_episode:
            return_value = True
        if target_task_score > self.target_task_success_threshold:
            print('success in find terminationn')
            self.student_success = True

        if args.reward_function == 'simple_LP' or args.reward_function == 'cost' or args.reward_function == 'L2T' or args.reward_function == 'LP':
            if target_task_score > self.target_task_success_threshold:
                print('success in find terminationn')
                return_value = True

        return return_value



    def get_LP(self, new_G, task_index, args):
        past_G = self.student_returns_dict.get(task_index) #LP dict stores the most recent student return on a particular env
        
        if past_G == None:
            past_G = self.min_reward 

        if args.normalize:
            print(f'task index = {task_index}')
            print('new G before normalizing', new_G)
            new_G = utils.normalize(-(args.max_time_step+1), self.get_max_value(task_index), new_G)
            print(f'new G after normalizing= {new_G} past G = {past_G}')
        
        LP = new_G-past_G #made a change here
        if args.normalize:
            print(f'LP = {LP}')

        #assert LP <= 1 
        if args.reward_function == 'simple_ALP':
            LP = abs(new_G-past_G)
            assert LP <= 1 
        
        reward = LP
        if args.debug:
            print('returns dict', self.student_returns_dict)
            print('past return', past_G, 'return now', new_G, 'LP', LP)
        
        return reward 



    def get_policy_buffer_state_rep(self, q_values, action, args):
    
        if args.student_type == 'PPO': #this is a strict one hot vector always
            one_hot_vector = utils.get_one_hot_action_vector(action, 1, 0, args.student_num_actions)
        else:
            if all(element == q_values[0] for element in q_values):
                one_hot_vector = [.25]*args.student_num_actions

            else:
                action = np.argmax(q_values)
                one_hot_vector = utils.get_one_hot_action_vector(action, 1, 0, args.student_num_actions)
        return one_hot_vector
    
    def get_embeb_q_value_state_rep(self, q_values, action, args):
        if args.student_type == 'q_learning' or args.student_type == 'sarsa':#This would embed the all q values of the state
            return q_values
        else: #this is really only used for ppo student
             #This would embed the q value of the argmax action
            return utils.get_one_hot_action_vector(action, q_values, 0, args.student_num_actions)
        
        #we could also think about embedding the q value of the sampled action taken in that state (might not be the argmax action)
    
    


    def add_obs_to_buffer(self,obss, q_values, actions, args):
    
        shape = np.shape(obss)
        

        for i in range(0, shape[0]): #this goes through all the observations encountered during a training episode(s)
            state = obss[i]
        
            if args.SR == 'buffer_policy':
                one_hot_vector = self.get_policy_buffer_state_rep(q_values[i], actions[i], args)
                #print('vector', one_hot_vector)

            #buffer q table returns the q values of all actions 
            if args.SR == 'buffer_q_table':
                one_hot_vector = self.get_embeb_q_value_state_rep(q_values[i], actions[i],args)
                #print('vector', one_hot_vector)

            #buffer action only returns the action the policy took
            if args.SR == 'buffer_action':
                one_hot_vector = actions[i]
       
            state = utils.convert_array_to_tuple(state, args)
 
      
            self.policy_table[state] = one_hot_vector
            self.student_state_buffer.append(state)

        #this makes sure we don't have the same state repeated 
        self.student_state_buffer = list(set(self.student_state_buffer))



           




#Note for callie: I can worry about this second.
#First I should just see if my code works with the newest updates.
#I still need to check on the seed stuff. 
#buffer q table for ppo is really the value function for the state
#buffer policy for ppo can actually take the max action, by using the value with the highest 
#prob according to the policy.. but I think what it currently does is take the sampled action according to the soft max
#buffer action is looking at the sampled action.
#I think we can do a buffer q table for fetch reach by looking at the q value of only the sampled action. 
#for the grid envs, for buffer policy, I should prob take the sampled action instead of the max..