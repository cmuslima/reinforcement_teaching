from training_loop import teaching_training
from evalution_loop import teacher_evaluation
import utils
import numpy as np
def run_training_loop(args):
    save_single_run = False
    if save_single_run:
        end = args.num_runs_start+1
    else:
        end = args.num_runs_end
  
    dir = f'{args.rootdir}/teacher-checkpoints'

    for seed in range(args.num_runs_start, end):

        print(f'\n\n\n\nrun {seed} with teacher SR = {args.SR } and batch size = {args.teacher_batchsize} and learning rate = {args.teacher_lr} buffer = {args.teacher_buffersize}, reward {args.reward_function}\n\n\n\n')
        file = utils.get_file_name(args, dir, seed)
        print(f'file being used = {file}')
        teacher_return_list, teacher_scores, teacher_actions, eval_student_score  = teaching_training(seed, args, file)
        print(f'Teacher returns on run {seed}, {teacher_return_list}')
    
    
        collected_data = [teacher_return_list, teacher_scores, teacher_actions, eval_student_score]
        data_names = ['teacher_return_list', 'teacher_scores', 'teacher_actions', 'eval_student_score']

        for idx, data in enumerate(collected_data):
            dir = f'{args.rootdir}/teacher-data'
            utils.make_dir(args, dir)
            file_details = [data_names[idx], seed]
            model_name = utils.get_model_name(args, dir, file_details)
            utils.save_data(model_name,data)
    print(f'run {seed} complete')
        
def run_evaluation_loop(args):
    
    teacher_scores = []
    teacher_actions = []
    student_scores = []
    save_all_runs = False
    save_single_run = True
    print(args.num_runs_start)
    print(args.num_runs_end)
    curr_data = []
    
    #try:
    if save_single_run:
        end = args.num_runs_start+1
    else:
        end = args.num_runs_end
    for seed in range(args.num_runs_start, end):

        print(f'run {seed} with student lr = {args.student_lr} with batch size = {args.teacher_batchsize} and learning rate = {args.teacher_lr} buffer = {args.teacher_buffersize}, reward {args.reward_function}')

        try:
            try:
                dir = f'{args.rootdir}/teacher-checkpoints'
                file = utils.get_file_name(args, dir, seed)        
                print(f'file being used = {file}')

                teacher_score, teacher_action_list, student_score = teacher_evaluation(args, file, seed)
                print('completed teacher eval')
                teacher_scores.append(teacher_score)
                teacher_actions.append(teacher_action_list)
                student_scores.append(student_score)
                
                print(f'teacher_action_list = {teacher_action_list}')
                
                print(f'student_score', student_score)
            except:
                dir = f'{args.rootdir}/teacher-data'
                file = utils.get_file_name(args, dir, seed)        
                print(f'file being used = {file}')

                teacher_score, teacher_action_list, student_score = teacher_evaluation(args, file, seed)
                print('completed teacher eval')
                teacher_scores.append(teacher_score)
                teacher_actions.append(teacher_action_list)
                student_scores.append(student_score)
                
                print(f'teacher_action_list = {teacher_action_list}')
                
                print(f'student_score', student_score)
        except:
            continue
    # except:
    #     print(f'File with batch size = {args.teacher_batchsize}, teacher lr = {args.teacher_lr}, buffer = {args.teacher_buffersize}, reward {args.reward_function}, SR = {args.SR} and seed {seed} DOES NOT EXIST')
    #     save_all_runs = False
    #     save_single_run = False
    if save_single_run:

        collected_data = [teacher_score, teacher_action_list, student_score]
        data_names = ['teacher_score', 'teacher_action_list', 'student_score']
        for idx, data in enumerate(collected_data):
            dir = f'{args.rootdir}/evaluation-data'
            if args.random_curriculum:
                dir = f'./RT/{args.env}/random_curriculum'
                print('should be here', dir)
            if args.target_task_only:
                dir =  f'./RT/{args.env}/target_task_only'
            if args.HER:
                dir = dir + '/HER'
                print('should be here b/c I am using HER')
            if args.multi_students:
                dir = f'{dir}/{args.student_type}' 
            utils.make_dir(args, dir)
            file_details = [data_names[idx], str(args.num_runs_start)]
            model_name = utils.get_model_name(args, dir, file_details)
            utils.save_data(model_name,data)
    if save_all_runs:
        assert len(student_scores) == args.num_runs
        try:
            raw_averaged_returns = [np.mean(np.array(student_scores), axis = 0), np.std(np.array(student_scores), axis = 0)]
            print('raw_averaged_return', raw_averaged_returns)
        except:
            raw_averaged_returns = 'all returns were 0'
        #print('raw_averaged_return', raw_averaged_returns)
        #print(f'teacher_action_list = {teacher_action_list}')
        #print(f'student_score', student_score)

        collected_data = [teacher_scores, teacher_actions, raw_averaged_returns]
        data_names = ['teacher_scores', 'teacher_actions', 'raw_averaged_returns']
        for idx, data in enumerate(collected_data):
            dir = f'{args.rootdir}/evaluation-data'
            if args.random_curriculum:
                dir = f'./RT/{args.env}/random_curriculum'
                print('should be here', dir)
            if args.target_task_only:
                dir =  f'./RT/{args.env}/target_task_only'
            if args.HER:
                dir = dir + '/HER'
                print('should be here b/c I am using HER')
            if args.multi_students:
                dir = f'{dir}/{args.student_type}' 
            utils.make_dir(args, dir)
            file_details = [data_names[idx], '']
            model_name = utils.get_model_name(args, dir, file_details)
            utils.save_data(model_name,data)
