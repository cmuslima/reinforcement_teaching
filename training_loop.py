
import numpy as np
from init_teacher import initialize_teacher_functions
from teacher_env import teacher_environment



def teaching_training(seed, args, file = None):

  
    eps = args.teacher_eps_start   
    print(f'teacher eps {eps}')  
    all_student_scores = []
    all_teacher_actions = []
    teacher_return_list = []
    teacher_agent_functions = initialize_teacher_functions(args)
    teacher_env = teacher_environment(args)
    teacher_env.build_teacher_env(args, seed)
    print(f'teacher_env.teacher_action_size {teacher_env.teacher_action_size}')
    teacher_agent = teacher_agent_functions.build_teacher(args, seed,teacher_env.teacher_state_size,teacher_env.teacher_action_size, file)
    print(f'Finished building the teacher env and the teacher agent')

    print('run = ', seed)
    #one reset here which initalizes the env, teacher state space, etc
    for teacher_episode in range(0, args.teacher_episodes):
      
        print('teacher episode', teacher_episode)      
        teacher_return = 0
        student_scores = list() #this keeps a list of student scores per each teacher episode.      
        teacher_action_list = list()

                                                                       
        task_index, traj = teacher_env.reset(args, teacher_agent_functions)
        teacher_action_list.append(task_index)
        print(f'Got my first teacher action {task_index}')
        
        for current_student_episode in range(1, args.student_episodes+1):
          
            print('student episode', current_student_episode)
            task_index, task_name = teacher_agent_functions.get_teacher_action(teacher_agent, args, teacher_env, traj, eps) # # loop N times
            print(f'Got teacher action with index {task_index} and name {task_name}')
            teacher_action_list.append(task_index)
            traj_prime, reward, done, target_task_score = teacher_env.step( task_index, task_name, current_student_episode, args)
            print(f'Teacher reward = {reward} done = {done}')
            student_scores.append(target_task_score) #I want to collect data on the target task.. to see if over time, it approves on the target task.
            print(f'Student score on target task {target_task_score}')
            teacher_return+=reward
            

            if args.training:
                teacher_agent.update(traj, task_index, reward, traj_prime, done, args) ##loop N times
                print(f'Just updated the teacher agent')
            
            traj = traj_prime
            eps = teacher_agent_functions.get_eps(args, eps)
            print('changing eps', eps)


            
    
            if done:
                if args.training:
                    teacher_return_list.append(teacher_return)
                    all_teacher_actions.append(teacher_action_list)
                    print(f'teacher return {teacher_return}')
                    print(f'teacher_action_list {teacher_action_list}')
                    all_student_scores.append(student_scores)
                    break
                else:
                    all_teacher_actions = teacher_action_list
                    all_student_scores = student_scores
                    continue
            #print('finished a student episode \n')
                
                

        teacher_agent_functions.save_teacher_data(args, teacher_agent, teacher_return_list, teacher_return, seed)

    return teacher_return_list, all_teacher_actions, all_student_scores


 
            
