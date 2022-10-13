
import utils
import numpy as np
def run_evaluation_loop(args):
    
    student_scores = []
    save_single_run = args.compute_canada 
    area_under_curve_list = []
    
    
    if save_single_run:
        end = args.num_runs_start+1
    else:
        end = args.num_runs_end
    print('START', args.num_runs_start)
    print('END',end)
    for seed in range(args.num_runs_start, end):

        print(f'run {seed} with student lr = {args.student_lr} with batch size = {args.teacher_batchsize} and learning rate = {args.teacher_lr} buffer = {args.teacher_buffersize}, reward {args.reward_function}')


        dir = f'{args.rootdir}/teacher-data'
        file = utils.get_file_name(args, dir, seed)        
        print(f'file being used = {file}')
      
        teaching_training = utils.import_training_modules(args)

        _, teacher_action_list, target_task_student_scores  = teaching_training(seed, args, file)

        area_under_single_curve = np.sum(target_task_student_scores)
        area_under_curve_list.append(area_under_single_curve)
        student_scores.append(target_task_student_scores)
        
        print(f'teacher_action_list = {teacher_action_list}')
        print(f'student_score', target_task_student_scores)
    
    


        collected_data = [ teacher_action_list, target_task_student_scores]
        data_names = ['teacher_action_list', 'target_task_student_scores']

        for idx, data in enumerate(collected_data):

            dir = f'{args.rootdir}/evaluation-data'
            utils.make_dir(args, dir)
            index = str(seed)

            file_details = [data_names[idx], index]
            model_name = utils.get_model_name(args, dir, file_details)

            utils.save_data(model_name,data)


    try:
        raw_averaged_returns = [np.mean(np.array(student_scores), axis = 0), np.std(np.array(student_scores), axis = 0)]
        print('raw_averaged_return', raw_averaged_returns)
    except:
        raw_averaged_returns = 'all returns were 0'

    if save_single_run == False:
        model_name = utils.get_model_name(args, dir, ['raw_averaged_returns', ''])
        utils.save_data(model_name,raw_averaged_returns)
        model_name = utils.get_model_name(args, dir, ['AUC', ''])
        utils.save_data(model_name,area_under_curve_list)   
        # if args.student_transfer:
        #     utils.make_dir(args, f'{args.rootdir}/evaluation-data/transfer_q_learning_to_sarsa')
        #     model_name = f'{args.rootdir}/evaluation-data/transfer_q_learning_to_sarsa/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_sarsa_{seed}'
        # if args.student_lr_transfer:
        #     utils.make_dir(args, f'{args.rootdir}/evaluation-data/lr_transfer_{args.student_lr}')
        #     model_name = f'{args.rootdir}/evaluation-data/lr_transfer_{args.student_lr}/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{args.student_lr}_{seed}'
        # if args.student_NN_transfer:
        #     utils.make_dir(args, f'{args.rootdir}/evaluation-data/NN_large_128_128')
        #     model_name = f'{args.rootdir}/evaluation-data/NN_large_128_128/student_score_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        
    
        

    




