requirements

numpy
scipy
torch
tensorflow==1.14 (this is b/c of baselines)
pip install torch-ac
gym==0.21.0 
tensorboardX
gym-minigrid==1.0.2


Automatic Curriculum Learning Experiments:


How to run a experiment:
1. Clone the repo
2. create a virtual env. 
    On Mac OS
    a. python -m venv virtual_env_name
    b. source virtual_env_name/bin/activate
3. Example training run for maze student env + RL teacher using the PE state rep and LP reward proposed in our RT paper.
 python3 config.py --env maze --teacher_agent DQN --reward_function simple_LP --SR buffer_policy --teacher_batchsize 128 --teacher_lr 0.001 --teacher_buffersize 100 --training 1 --evaluation 0 --num_runs_start 0 --num_runs_end 10

The config.py file contains all the teacher/student/experiment hyperparameters.
There is a hyper_param_sweep bool in the config file. There is a loop that loops over various teacher configurations. If this is set to True, then in the config file, you can edit what values you want to loop. For example, what teacher lrs, teacher state represetnations, etc. 
Once you edit that, you can run python config.py --env env_name --teacher_agent DQN --num_runs_start 0 --num_runs_end 1 --training True --evaluation False

4.  Example evaluation run for maze student env + RL teacher using the PE state rep and LP reward proposed in our RT paper. 
 python3 config.py --env maze --teacher_agent DQN --reward_function simple_LP --SR buffer_policy --teacher_batchsize 128 --teacher_lr 0.001 --teacher_buffersize 100 --training 0 --evaluation 1 --num_runs_start 0 --num_runs_end 10 --trained_teacher 1 --hyper_param_sweep 0


How to get the teacher’s action for the student:
get_teacher_action function in init_teacher.py

How to change teacher’s action if the teacher assigns the same action twice in a row:
change_teacher_acton in init_teacher.py

How to change the maze to have different blocked states:
Go to basic_grids() class in grid_envs.py

You can changed the hard coded start_state and termination_state.You can also change the hard coded blocked states list.

How to create the heat maps:
create_curriculum_heat_maps file


