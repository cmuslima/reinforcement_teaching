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

def plot_actions_regions(args):


    interval1 = []
    interval2 = []
    interval3 = []
    positions = dict()
    
    count = 0
    for i in range(0, args.rows):
        for j in range(0, args.columns):
            position = [i, j]
            position = np.array(position)
            positions.update({str(position): [0,0,0]})
            count+=1
    beginning = 5
    middle = 10
    total_count = [0,0,0]
    
    for seed in range(0,args.num_runs):
        file = f'{args.rootdir}/evaluation-data/teacher_action_list_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_{seed}'
        data = utils.get_data(file)
        data_first = data[0:beginning]
        data_middle = data[beginning:middle]
        data_end = data[middle:]
        
        for idx, value in enumerate(data_first):
            print(idx, value)
            if idx == 0:
                value = (7,np.array([2,12]))
            positions[str(value[1])][0] +=1
            total_count[0]+=1
        for idx, value in enumerate(data_middle):
            positions[str(value[1])][1] +=1
            total_count[1]+=1
        for idx, value in enumerate(data_end):
            positions[str(value[1])][2] +=1
            total_count[2]+=1
            
    
    for i in range(3):
        probability = []
        for r in range(0, args.rows):
            for c in range(0, args.columns):
                position = np.array([r, c])        
                probability.append(positions[str(position)][i]/total_count[i])
        #print(raw_freqs, probability)
        print('rendering')
        render_env(probability)
    
