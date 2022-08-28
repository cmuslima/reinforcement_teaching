
import numpy as np
import utils
from student_env_setup import build_env

class train_evaluate_protocol():
    def __init__(self, args):
        if args.env == 'fetch_push':
            self.run, self.build_env = utils.import_modules(args)
        if args.env == 'fetch_reach_3D_outer':
            self.RT_run  = utils.import_modules(args)
        if args.env == 'four_rooms' and args.tabular == False:
            print('loading four rooms modules')
            self.student_train, self.evaluate_task = utils.import_modules(args)
        if args.env == 'maze' or args.env == 'cliff_world':
            self.env = build_env(args)
            

        

    def train(self, student_agent, student_type, task_name, args, model, env_dict, config_params, dims):

        if args.tabular:
            params = None
            model = None
            score, obss, q_values, actions = self.train_tabular(student_agent, task_name, args, student_type)
 
        if args.student_type == "PPO":
            params = None
            model = None
            score, obss, q_values, actions = self.student_train(args, task_name)
            

        if args.student_type == 'DDPG':
            if args.env == 'fetch_reach_3D_outer':
                model, obss, actions, params, config_params, dims = self.RT_run.student_training(model, task_name, args, env_dict, False, config_params, dims)
                q_values = None
                score = None

            if args.env == 'fetch_push':
            
                training_complete = False
                model, obss, actions, params, config_params, dims = self.run.student_training(model, config_params, dims, training_complete, env_dict, task_name, args)
                q_values = None
                score = None
        return score, obss, q_values, actions, model, params, config_params, dims 
    def visualize(self, args):

 
        self.visualize(args)
        
        #print('average_score', average_score)
        return 
    def evaluate(self, student_agent, task_name, args, model, env_dict, config_params, dims):

        if args.tabular:
            average_score, _, params = self.evaluate_task_tabular(student_agent, task_name, args)
        if args.student_type == 'DDPG':
            if args.env == 'fetch_reach_3D_outer':
                average_score, _, _  = self.RT_run.student_evalution(args, model, task_name, env_dict, config_params, dims)
                params = None
          
            if args.env == 'fetch_push':
                average_score = self.run.evaluation(model, task_name, env_dict, config_params, dims, args)
                params = None        
        if args.student_type == 'PPO':
            print('using a PPO student')
            average_score, params = self.evaluate_task(args, task_name)
        
    
        return average_score, params

    def source_target_evaluation(self, student_id,student_agent,task_name, target_task, args,  model, env_dict, config_params, dims):
        #print('STUDENT ID', student_id)
        source_task_score, _= self.evaluate(student_agent, task_name, args, model, env_dict, config_params, dims)
        target_task_score, params = self.evaluate(student_agent,target_task,args,  model, env_dict, config_params, dims)
        return source_task_score, target_task_score,params


    def evaluate_task_tabular(self, student_agent, task_name, args): #this will evaluate it for one episode 
        #print('new evaluation')
        score_per_episode = list()
        num_steps_per_episode = list()

       
        eps = 0

        #print('ss', ss)
        
        for i in range(args.num_evaluation_episodes):
            start_state = task_name #target task_name
            #print('start state', start_state)
            state = student_agent.reset(start_state)
            #print('state', state)
            self.env.reset(start_state)
            #print('termination state', self.env.termination_state)
            action_movement, action_index = student_agent.e_greedy_action_selection(state, self.env, eps)
            num_time_steps = 0
            score=0
            #print('first state', state)
            while(True):
                #print(student_agent.q_matrix[tuple(state)])
                next_state, reward, done = student_agent.step(state, action_index, action_movement, num_time_steps, self.env)
                score+=reward
                #print(f'S = {state} A = {action_index} R = {reward} S prime = {next_state}')
                #print(student_agent.q_matrix[(4,13)])
                if done or num_time_steps>=args.max_time_step:
                    num_time_steps+=1

                    score_per_episode.append(score)

                    num_steps_per_episode.append(num_time_steps) 

                    break
                state=next_state
                action_movement, action_index = student_agent.e_greedy_action_selection(state, self.env, eps)
                num_time_steps+=1

        assert len(score_per_episode) == args.num_evaluation_episodes
        average_score = np.mean(np.array(score_per_episode))
        average_time_step = np.mean(np.array(num_steps_per_episode))

        return average_score, average_time_step, list(student_agent.q_matrix) #student_agent.state_buffer, q_values, student_agent.action_buffer


    def train_tabular(self, student_agent, task_name, args, student_type):
        #print('inside train')
        #print(student_agent.q_matrix)
        student_agent.clear_buffer()
        #print('student agent buffer', student_agent.state_buffer, student_agent.action_buffer)
        average_score = []
        for i in range(args.num_training_episodes):
        
            state = student_agent.reset(start_state = task_name)
            ##print('state', state)
            self.env.reset(task_name)
            #print('start state', self.env.start_state)
            #print('goal state', self.env.termination_state)
            action_movement, action_index = student_agent.act(state, self.env)
            #print(f'first state = {state} first action = {action_index}')
            score = 0
            num_time_steps = 0
            while(True):

        
            
                next_state, reward, done = student_agent.step(state, action_index, action_movement, num_time_steps, self.env)
                score += reward


                if num_time_steps>=args.max_time_step:
                    #print(num_time_steps)
                    done = True
                        
                next_action_movement, next_action_index = student_agent.act(next_state, self.env)
                #print(f' state = {next_state} action = {next_action_index}')
                if student_type == 'sarsa':
                    #print(f'making sarsa update')
                    if args.student_transfer:
                        student_agent.q_learning_update(state, action_index, reward, next_state, done)
                    else:
                        student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                
                if student_type == 'q_learning':
                    #print(f'making q learning update')
                    if args.student_transfer:
                        student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                    else:
                        #print('updating with', state, action_index, reward, next_state)
                        student_agent.q_learning_update(state, action_index, reward, next_state, done)
                if done:
                    num_time_steps+=1
                    average_score.append(num_time_steps)                              
                    break
                
                action_index = next_action_index
                action_movement = next_action_movement
                state=next_state
                num_time_steps+=1


        

        assert len(student_agent.state_buffer) == len(student_agent.action_buffer)
        
        q_values = utils.get_q_values(student_agent.state_buffer, student_agent.action_buffer, student_agent)

        average_score = np.mean(average_score)
        return score, student_agent.state_buffer, q_values, student_agent.action_buffer  # need to get these

