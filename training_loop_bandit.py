
import numpy as np
from init_teacher import teacher_utils
import init_student




def teaching_training(seed, args, file, return_list):


    done = False
    teacher_help_fns = teacher_utils(args)
    teacher_agent, student_env = teacher_help_fns.initalize_teacher(args, seed, None, file)
    #train_evalute_protocol(student_env)
    #student_env will be None for non-tabular student agents

    #print(f'configration: environment = {args.env}, reward function type: {rf}, number of tasks = {env_config.teacher_action_size}, alpha = {args.alpha}, batch size = {b}, step size = {lr} buffer = {buffer}')
    
    print('run = ', seed)
    #one reset here which initalizes the env, teacher state space, etc
    student_scores = list()
    teacher_action_list = list()
    student_agent = init_student.initalize_single_student(args)
    teacher_help_fns.reset_teacher_params(args, args.student_type) 
    print('student_episodes', args.student_episodes)
    for i_episode in range(1, args.student_episodes+1):

            

        task_index, task_name = teacher_help_fns.get_teacher_action(teacher_agent,args) # # loop N times
        print('task index', task_index, 'task_name', task_name)
        
        teacher_action_list.append(task_index)
       


        source_task_score, target_task_score, source_task_training_score = teacher_help_fns.student_train_protocol(student_agent, task_name, args) # loop 1 times, with N processes


        LP = teacher_help_fns.get_LP(source_task_score, task_index, args) #loop N times 

        teacher_help_fns.update_teacher_dicts(task_index, source_task_score, target_task_score, LP, i_episode, args) #loop N times 


        reward = teacher_help_fns.get_teacher_reward(task_index, LP, target_task_score,source_task_training_score, i_episode, args) ##loop N times 

        print(f'TEACHER ACTION = {task_index}, TEACHER REWARD = {reward}')
        teacher_agent.update_action_prob(reward, task_index, args)

        
        _ = teacher_help_fns.find_termination(target_task_score, i_episode, args.student_episodes, args)
        if i_episode == args.student_episodes:
            done = True
    
        teacher_help_fns.evaluation_true(args, i_episode)
    
        student_scores.append(target_task_score) #I want to collect data on the target task.. to see if over time, it approves on the target task.
   
        if done:
            if args.student_type == 'DDPG':
                teacher_help_fns.close_run(args)
            break
       
            



    print('teacher actions',teacher_action_list)
    print('student scores', student_scores)
    return None, None, teacher_action_list, student_scores


 
            
