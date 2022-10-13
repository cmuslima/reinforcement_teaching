import pickle
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy import stats
from plotting_graphs import get_data, calculate_area_under_curve
import pickle
import utils
import matplotlib.pyplot as plt


def quick_average_data(args, num_runs, name):
    averaged_data = []
    area_under_curve = 0
    area_under_curve_list = []
    for i in range(0, num_runs):
        data = get_data(f'{name}_{i}')
        #print('data', data)
        area_under_single_curve = np.sum(data)
        #print('area under curve', area_under_single_curve)
        averaged_data.append(data)
        area_under_curve_list.append(area_under_single_curve)

    averaged_data = [np.mean(np.array(averaged_data), axis = 0), np.std(np.array(averaged_data), axis = 0)]
    print('averaged_data', averaged_data)
    print('area_under_curve', np.mean(area_under_curve_list))
    return averaged_data, area_under_curve_list

def average_all(args):
    learning_rates = [.001, .005]
    buffer_sizes = [75,100]
    batch_sizes = [128,256,64] 
    all_AUC_dict = {}

    for lr in learning_rates:
        for buffer in buffer_sizes:
            for batch in batch_sizes:     
                #for student_lr in student_lrs:
                #args.student_lr = student_lr  
                args.teacher_batchsize = batch
                args.teacher_lr = lr
                args.teacher_buffersize = buffer

                args.rootdir = utils.get_rootdir(args, args.SR)
                print('args.rootdir on config', args.rootdir)

                
                if args.setting == 'RL':
                    if 'buffer' in args.SR:
                        model_name_averaged_data = f'{args.rootdir}/evaluation-data/{args.SR}_{args.reward_function}_{lr}_{batch}_{buffer}_raw_averaged_returns'
                        model_name_AUC = f'{args.rootdir}/evaluation-data/{args.SR}_{args.reward_function}_{lr}_{batch}_{buffer}_AUC'

                        file_name = f'{args.rootdir}/evaluation-data/target_task_student_scores_{args.SR}_{args.reward_function}_{lr}_{batch}_{buffer}'
                    
                    else:
                        model_name_averaged_data = f'{args.rootdir}/evaluation-data/{args.SR}_{args.reward_function}_{lr}_{batch}_raw_averaged_returns'
                        model_name_AUC = f'{args.rootdir}/evaluation-data/{args.SR}_{args.reward_function}_{lr}_{batch}_AUC'
                        file_name = f'{args.rootdir}/evaluation-data/target_task_student_scores_{args.SR}_{args.reward_function}_{lr}_{batch}'

                averaged_data, area_under_curve = quick_average_data(args, args.num_runs,file_name )
                all_AUC_dict.update({model_name_AUC: np.sum(area_under_curve)})
                
                
                print('model name', model_name_averaged_data)
                print('averaged_data', averaged_data)
                print('area_under_curve', area_under_curve)
                utils.save_data(model_name_averaged_data,averaged_data)
                utils.save_data(model_name_AUC,area_under_curve)
    
    max_key_AUC = max(all_AUC_dict, key=all_AUC_dict.get)
    print('max_key_AUC', max_key_AUC)
    max_key_returns = max_key_AUC[:-4]
    max_key_returns = f'{max_key_returns}_raw_averaged_returns'
    print('max_key_returns', max_key_returns)
    print('max_key_AUC', max_key_AUC, 'max_value', all_AUC_dict[max_key_AUC])
                
    max_key_AUC_list = [utils.get_data(max_key_AUC), utils.get_data(max_key_returns), max_key_AUC, max_key_returns]
    
    
    utils.save_data(f'{args.rootdir}/{args.reward_function}_MAX_AUC_data',max_key_AUC_list)

    print('max_key_AUC_list', max_key_AUC_list)