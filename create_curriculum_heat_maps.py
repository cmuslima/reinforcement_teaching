import utils
from rendering_envs import render_env
def plot_actions(args):

    #may need to change the file name 
    file = f'{args.rootdir}/evaluation-data/teacher_actions_{args.SR}_{args.reward_function}_{args.teacher_lr}_{args.teacher_batchsize}_{args.teacher_buffersize}_'
    print(file)
    data = utils.get_data(file)
    print(len(data))
    print(len(data[0]))
    interval1 = []
    interval2 = []
    interval3 = []
    bad_count = 0
    total_count = 0 
    beginning = 5
    middle = 10
    #this assumes that all the curriculums (i.e. one per run/seed are stored in a single collective file) which is how I have the data stored
    for seed in range(0,args.num_runs):
        curr = data[seed]
    
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
