#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.minigrid import Grid
# from gym_minigrid.register import register
debug = False
from gym.envs.registration import register as gym_register



env_list = []

def register(
    id,
    entry_point,
    reward_threshold=0.95
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        reward_threshold=reward_threshold
    )

    # Add the environment to the set
    env_list.append(id)

class SimpleFourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """
    

    def __init__(self, agent_pos, goal_pos):
        if debug:
            print(f'agent pos', agent_pos)
        self._agent_default_pos = agent_pos #np.array([3,3])
        self._goal_default_pos = goal_pos #np.array([10,3])
        #print(f'agent pos', self._agent_default_pos)
        print('inside init simple four rooms', self._agent_default_pos, self._goal_default_pos)
        super().__init__(grid_size=9, max_steps=40)
        print('do I pass this super')

    #def place_agent(self, agent_pos, goal_pos):
        # self._agent_default_pos = agent_pos
        # self._goal_default_pos = goal_pos

    # def _gen_grid(self, width, height):
    #     # Create an empty grid
    #     self.grid = Grid(width, height)

    #     # Generate the surrounding walls
    #     self.grid.wall_rect(0, 0, width, height)

    #     # Place a goal in the bottom-right corner
    #     self.put_obj(Goal(), width - 2, height - 2)

    #     # Create a vertical splitting wall
    #     splitIdx = self._rand_int(2, width-2)
    #     self.grid.vert_wall(splitIdx, 0)

    #     # Place the agent at a random position and orientation
    #     # on the left side of the splitting wall
    #     self.place_agent(size=(splitIdx, height))

    #     # Place a door in the wall
    #     doorIdx = self._rand_int(1, width-2)
    #     self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

    #     # Place a yellow key on the left side
    #     self.place_obj(
    #         obj=Key('yellow'),
    #         top=(0, 0),
    #         size=(splitIdx, height)
    #     )
    def _gen_grid(self, width, height):
        print('am I here')
        # Create the grid
        self.grid_x = Grid(width, height)
        print('self.grid.max_steps', self.grid_x.width)
        print('self.grid', self.grid_x.grid)
        # Generate the surrounding walls
        #self.grid.horz_wall(0, 0) #y is the fixed column 
        #self.grid.horz_wall(0, height - 1)
        #self.grid.vert_wall(0, 0)
        #self.grid.vert_wall(width - 1, 0)
        self.grid_x.printing()
        outer_walls = []
        for i in range(0,9):
            outer_walls.append((0, i))
            outer_walls.append((8, i))
        for i in range(1,8):
            outer_walls.append((i, 0))
            outer_walls.append((i, 8))
        self.blocked_states= outer_walls + [(4,1), (1,4), (7,4), (4,7), (3,4), (5,4), (4,3), (4,4), (4,5)]


        for state in self.blocked_states:
            self.grid.place_wall(state[0], state[1])

        # room_w = width // 2
        # room_h = height // 2

        # For each row of rooms
        # for j in range(0, 2):

        #     # For each column
        #     for i in range(0, 2):
        #         xL = i * room_w
        #         yT = j * room_h
        #         xR = xL + room_w
        #         yB = yT + room_h

        #         # Bottom wall and door
        #         if i + 1 < 2:
        #             self.grid.vert_wall(xR, yT, room_h)
        #             pos = (xR, self._rand_int(yT + 1, yB))
        #             self.grid.set(*pos, None)

        #         # Bottom wall and door
        #         if j + 1 < 2:
        #             self.grid.horz_wall(xL, yB, room_w)
        #             pos = (self._rand_int(xL + 1, xR), yB)
        #             self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        #if self._agent_default_pos is not None:
        self.agent_pos = self._agent_default_pos
        if debug:
            print(f'self.agent_pos', self.agent_pos)
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        #else:
            #self.place_agent()

        #if self._goal_default_pos is not None:
        goal = Goal()
        self.put_obj(goal, *self._goal_default_pos)
        goal.init_pos, goal.cur_pos = self._goal_default_pos
        #else:
            #self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


#weird bug in the goal_pos where the x, y goal needs to be y,x in order to display correctly
class Simple4rooms0(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([1,1])
        goal_pos = np.array([5,6])
        print('inside simple4fours init')
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class Simple4rooms1(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([2,1])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class Simple4rooms2(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([2,2])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class Simple4rooms3(SimpleFourRoomsEnv):
    def __init__(self):
        #agent_pos = np.array([6,4])
        agent_pos = np.array([2,4]) # mini point 
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class Simple4rooms4(SimpleFourRoomsEnv):
    def __init__(self):
        
        #agent_pos = np.array([2,4]) #top corrdior
        #agent_pos = np.array([6,2])
        agent_pos = np.array([2,6])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)


class Simple4rooms5(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([4,6]) # right corridor
        goal_pos = np.array([5,6])
        print('inside simple4fours init')
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class Simple4rooms6(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([6,6]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class Simple4rooms7(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([6,4]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class Simple4rooms8(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([6,2]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class Simple4rooms9(SimpleFourRoomsEnv):
    def __init__(self):
        agent_pos = np.array([4,2]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

register(
    id='MiniGrid-Simple-4rooms-0-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms0'
)

register(
    id='MiniGrid-Simple-4rooms-1-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms1'
)
register(
    id='MiniGrid-Simple-4rooms-2-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms2'
)

register(
    id='MiniGrid-Simple-4rooms-3-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms3'
)

register(
    id='MiniGrid-Simple-4rooms-4-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms4'
)

register(
    id='MiniGrid-Simple-4rooms-5-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms5'
)
register(
    id='MiniGrid-Simple-4rooms-6-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms6'
)

register(
    id='MiniGrid-Simple-4rooms-7-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms7'
)

register(
    id='MiniGrid-Simple-4rooms-8-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms8'
)
register(
    id='MiniGrid-Simple-4rooms-9-v0',
    entry_point='rl_starter_files_master.gym_minigrid.envs.simple4rooms:Simple4rooms9'
)


# register(
#     id='MiniGrid-Simple-FourRooms-v0',
#     entry_point='gym_minigrid.envs.simple4rooms:SimpleFourRoomsEnv'
    
# )
