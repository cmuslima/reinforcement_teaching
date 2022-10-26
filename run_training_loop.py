

import utils
import numpy as np
def run_train_loop(args):
    save_single_run = args.compute_canada
    if save_single_run:
        end = args.num_runs_start+1
    else:
        end = args.num_runs_end
  
    dir = f'{args.rootdir}/teacher-data'

    for seed in range(args.num_runs_start, end):


        print(f'\n\n\n\nrun {seed} with teacher SR = {args.SR} and batch size = {args.teacher_batchsize} and learning rate = {args.teacher_lr} buffer = {args.teacher_buffersize}, reward {args.reward_function}\n\n\n\n')
        print(f'start seed = {args.num_runs_start}, ending seed = {end}')

        
        teaching_training = utils.import_training_modules(args)
        print(f'{args.expert_teacher_model_file_name}_{seed}.pth')

        
        teacher_return_list, teacher_actions, target_task_student_scores  = teaching_training(seed, args, f'{args.expert_teacher_model_file_name}_{seed}.pth')


        print(f'Teacher returns on run {seed}, {teacher_return_list}')
    
    
        collected_data = [teacher_return_list, teacher_actions, target_task_student_scores]
        data_names = ['teacher_return_list', 'teacher_actions', 'target_task_student_scores']

        for idx, data in enumerate(collected_data):
            dir = f'{args.rootdir}/teacher-data'
            utils.make_dir(args, dir)
            file_details = [data_names[idx], seed]
            model_name = utils.get_model_name(args, dir, file_details)
            utils.save_data(model_name,data)


        
    print(f'run {seed} complete')
        


