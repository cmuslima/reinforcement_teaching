import utils
from rendering_envs import render_env
import numpy as np
# def plot_actions(args):

#     #may need to change the file name 
#     file = f'{args.rootdir}/evaluation-data/teacher_actions_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_'
#     print(file)
#     data = utils.get_data(file)
#     print(len(data))
#     # print(len(data[0]))
#     interval1 = []
#     interval2 = []
#     interval3 = []

#     beginning = 5
#     middle = 10
#     #this assumes that all the curriculums (i.e. one per run/seed are stored in a single collective file)
#     #otherwise you just have to loop through each file seperately
#     for seed in range(0,args.num_runs):
#         curr = data[seed]
    
#         interval1.append(curr[0:beginning]) #this looks at the first X number of elements in the teacher's curriculum
#         interval2.append(curr[beginning:middle]) 
#         interval3.append(curr[middle:])

#     flat_interval1 = [item for sublist in interval1 for item in sublist]
#     flat_interval2 = [item for sublist in interval2 for item in sublist]
#     flat_interval3 = [item for sublist in interval3 for item in sublist]
#     all_intervals = [flat_interval1, flat_interval2, flat_interval3]
  
#     for i in all_intervals:
#         raw_freqs = np.bincount(i)
#         probability = raw_freqs/sum(raw_freqs)
#         #print(raw_freqs, probability)
#         render_env(probability)
#     return probability


def plot_actions(args):


    interval1 = []
    interval2 = []
    interval3 = []

    beginning = 5
    middle = 10

    for seed in range(0,args.num_runs):
        file = f'{args.rootdir}/evaluation-data/teacher_action_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        data = utils.get_data(file)
        curr = []
        print(data)
        print(type(data))
       
        for idx, value in enumerate(data):
            #print(data)
            #print(idx, value)
            if idx == 0:
                curr.append(value)
            else:
                curr.append(value[0])
                
    #     curr = data[seed]
    
        interval1.append(curr[0:beginning]) #this looks at the first X number of elements in the teacher's curriculum
        interval2.append(curr[beginning:middle]) 
        interval3.append(curr[middle:])

    flat_interval1 = [item for sublist in interval1 for item in sublist]
    flat_interval2 = [item for sublist in interval2 for item in sublist]
    flat_interval3 = [item for sublist in interval3 for item in sublist]
    all_intervals = [flat_interval1, flat_interval2, flat_interval3]
  
    for i in all_intervals:
        raw_freqs = np.bincount(i)
        probability = raw_freqs/sum(raw_freqs)
        #print(raw_freqs, probability)
        render_env(probability)
    return probability