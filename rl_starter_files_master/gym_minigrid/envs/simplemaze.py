#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
debug = False

class SimpleMazeEnv(MiniGridEnv):
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
        super().__init__(grid_size=9, max_steps=40)

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
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        #self.grid.horz_wall(0, 0) #y is the fixed column 
        #self.grid.horz_wall(0, height - 1)
        #self.grid.vert_wall(0, 0)
        #self.grid.vert_wall(width - 1, 0)
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
class SimpleMaze0(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([1,1])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze1(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([2,1])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze2(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([3,1])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze3(SimpleMazeEnv):
    def __init__(self):
        #agent_pos = np.array([6,4])
        agent_pos = np.array([5,1]) # mini point 
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)


class SimpleMaze4(SimpleMazeEnv):
    def __init__(self):

        agent_pos = np.array([6,1])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)


class SimpleMaze5(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([7,1]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze6(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([1,2]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze7(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([2,2]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze8(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([3,2]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze9(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([4,2]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)



class SimpleMaze10(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([5,2])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze11(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([6,2])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze12(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([7,2])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze13(SimpleMazeEnv):
    def __init__(self):
        #agent_pos = np.array([6,4])
        agent_pos = np.array([1,3]) # mini point 
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze14(SimpleMazeEnv):
    def __init__(self):

        agent_pos = np.array([2,3])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)


class SimpleMaze15(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([3,3]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze16(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([5,3]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze17(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([6,3]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze18(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([7,3]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze19(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([2,4]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)



class SimpleMaze20(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([6,4])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze21(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([1,5])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze22(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([2,5])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze23(SimpleMazeEnv):
    def __init__(self):
        #agent_pos = np.array([6,4])
        agent_pos = np.array([3,5]) # mini point 
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze24(SimpleMazeEnv):
    def __init__(self):

        agent_pos = np.array([5,5])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)


class SimpleMaze25(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([6,5]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze26(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([7,5]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze27(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([1,6]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze28(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([2,6]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze29(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([3,6]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)



class SimpleMaze30(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([4,6])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze31(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([5,6])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze32(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([6,6])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze33(SimpleMazeEnv):
    def __init__(self):
        #agent_pos = np.array([6,4])
        agent_pos = np.array([7,6]) # mini point 
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

class SimpleMaze34(SimpleMazeEnv):
    def __init__(self):

        agent_pos = np.array([1,7])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)


class SimpleMaze35(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([2,7]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze36(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([3,7]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze37(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([5,7]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze38(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([6,7]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
class SimpleMaze39(SimpleMazeEnv):
    def __init__(self):
        agent_pos = np.array([7,7]) # right corridor
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)

def mass_register():
    for i in range(0,40):
        register(
            id=f'MiniGrid-Simple-maze-{i}-v0',
            entry_point=f'gym_minigrid.envs.simplemaze:SimpleMaze{i}'
        )


mass_register()
# register(
#     id='MiniGrid-Simple-FourRooms-v0',
#     entry_point='gym_minigrid.envs.simple4rooms:SimpleFourRoomsEnv'
    
# )
