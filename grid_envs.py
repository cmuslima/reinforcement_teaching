import numpy as np


class cliff_world():
    def __init__(self):       
        self.start_state= np.array([3,0])  #target start state   
        self.termination_state= np.array([3,11])    
        self.blocked_states = []
        for i in range(1,11):
            self.blocked_states.append((3, i))

        self.rows=4
        self.columns=12
        self.oneDmaze = False
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]
        self.buffer = []
        #print('inside the cliff world class')
    def reset(self, start):
        self.start_state = start
        #self.termination_state = goal

    def check_reward(self, state):
        #print('state in check reward', state)
        #print('self.termination_state', self.termination_state)
        if np.array_equal(state, self.termination_state) == True:
            reward = -1
            #print('won')
            terminal= True
        elif tuple(state) in self.blocked_states:
            reward = -100
            #print('fell in lava')
            terminal = True

        else:
            reward = -1
            terminal= False

        return reward, terminal

    def check_state(self, next_state, state, action):
        print()
        next_state_tuple= tuple(next_state)
        print('next_state_tuple', next_state_tuple)
        if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            next_state = state

        if (next_state_tuple[0]) > self.rows or (next_state_tuple[1] > self.columns) or (next_state_tuple[0] < 0) or (next_state_tuple[1] < 0):
            next_state = state
        return next_state  
    def step(self, state, action_index): #should return next_state, reward, done
        
        action_movement = self.action_list[action_index][0]
        next_state= action_movement + state #this applies the action and moves to the next state 
        reward, done = self.check_reward(next_state)
        

        next_state = self.check_state(next_state, state, action_index) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state
     
        return next_state, reward, done


class basic_grids():
    def __init__(self, env_type, columns, rows): 
        if env_type == 'maze':      
            self.start_state= np.array([10,4])  #target start state   
            self.termination_state= np.array([0,15])    
            self.blocked_states= [(0,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3), (10,3), (3,0), (3,1), (3,2), (7,0), (7,1), (7,2), (3,4), (3,5), (7,4), (7,6), (7,7), (3,7), (4,7), (5,7), (6,7), (8,7), (9,7), (10,7), (0,6), (1,6), (6,8), (6,9), (6,10), (0,10), (1,10), (2,10), (3,10),(8,10), (9,10),(10,10), (5,12), (5,13), (5,14), (5,15), (6,12), (7,12), (8,13), (8,14), (8,15)]
        
        if env_type == 'open_maze':
            self.start_state= np.array([10,4])  #target start state   
            self.termination_state= np.array([0,15])    
            self.blocked_states= [(0,3), (2,3), (3,3), (5,3), (6,3), (7,3), (9,3), (10,3), (3,0), (3,1), (3,2), (7,0), (7,1), (7,2), (3,4), (3,5), (7,4), (7,6), (7,7), (3,7), (4,7), (5,7), (6,7), (9,7), (10,7), (0,6), (1,6), (6,8), (6,9), (6,10), (0,10), (1,10), (2,10), (3,10),(8,10), (9,10),(10,10), (5,12), (5,13), (5,14), (5,15), (6,12), (8,13), (8,14), (8,15)]
        self.actions = list()
        for i in range(0, rows):
            for j in range(0, columns):
                action = [i, j]
                if action not in self.blocked_states:
                    action = np.array(action)
                    self.actions.append(action)


        
        self.rows=rows
        self.columns=columns
        self.oneDmaze = False
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3

        self.up3=np.array([-4,0])  #0
        self.down3=np.array([4, 0]) # 1
        self.left3=np.array([0, -4]) # 2
        self.right3=np.array([0, 4])  #3

        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]
        self.region_list=[(self.up3, 0),(self.down3, 1), (self.left3,2), (self.right3, 3)]
        self.state_buffer = []
        self.action_buffer = []
        self.student_discount = .99

    def reset(self, start):
        self.start_state = start
        #print(f'using start state {self.start_state}')
        #self.termination_state = goal
        #return self.get_state_rep(self.start_state, self.start_state,self.termination_state), start

    def get_state_rep(self, state, starting_state, goal_state):
        state_rep = np.zeros((self.rows, self.columns, 3))
        r = state[0]
        c = state[1]
        #print(f'row = {r} column = {c}')
        state_rep[r][c][0] = 1 #this indicates the state the student is currently at
        state_rep[starting_state[0]][starting_state[1]][1] = 1 #this indicates the start state of the task
        state_rep[goal_state[0]][goal_state[1]][2] = 1 #this indicates the goal state 
        
        #print(f'goal state = {goal_state}, starting state = {starting_state}')
        state_rep = np.reshape(state_rep, (1, self.rows*self.columns*3))
        return state_rep
    def check_reward(self, state, num_time_steps):
        sparse = True
        if sparse == True:
            if np.array_equal(state, self.termination_state) == True:
                reward = 1
                reward = self.student_discount**num_time_steps
                terminal= True
            else:
                reward = 0
                terminal= False
        else:
            if np.array_equal(state, self.termination_state) == True:
                reward = -1
                terminal= True
            else:
                reward = -1
                terminal= False

        return reward, terminal

    def check_state(self, next_state, state, action):
        
        next_state_tuple= tuple(next_state)
        #print('next state tuple', next_state_tuple)
        if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            #print('self.rows', self.rows, next_state_tuple[0])
            #print('columns', self.columns, next_state_tuple[1])
            #print('next state touple', next_state_tuple)
            #print('next state is invalid b/c the next state is either the rows, columns are out of bounds')
            #print('next state tuple', next_state_tuple)
            next_state = state

        if next_state_tuple in self.blocked_states: # bc if this happens it wouldnt be near a blocked state
            #print('blocked state is the issue') 
            #print('next state tuple', next_state_tuple)
            next_state = state #return back to start state
    
        if (next_state_tuple[0]) > self.rows or (next_state_tuple[1] > self.columns) or (next_state_tuple[0] < 0) or (next_state_tuple[1] < 0):
            # print(next_state_tuple[0],'> ', self.rows)
            # print(next_state_tuple[1], '>', self.columns)
            # print(next_state_tuple[0], '< 0')
            # print(next_state_tuple[1], '<0')
            #print('next state tuple', next_state_tuple)
            #print('next state is invalid b/c the next state is either the rows, columns are out of bounds')

            next_state = state
      
        return next_state 
    def step(self, state, action_index, num_time_steps): #should return next_state, reward, done
        
        action_movement = self.action_list[action_index][0]
        next_state= action_movement + state #this applies the action and moves to the next state 
        reward, done = self.check_reward(next_state, num_time_steps)
        

        next_state = self.check_state(next_state, state, action_index) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state
        grid_next_state = next_state
        next_state = self.get_state_rep(next_state, self.start_state, self.termination_state)
        return next_state, grid_next_state, reward, done
