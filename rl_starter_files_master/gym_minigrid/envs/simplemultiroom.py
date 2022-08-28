from gym_minigrid.minigrid import *
from gym_minigrid.register import register
debug = False
# class SimpleRoom:
#     def __init__(self,
#         top,
#         size,
#         entryDoorPos,
#         exitDoorPos
#     ):
#         self.top = top
#         self.size = size
#         self.entryDoorPos = entryDoorPos
#         self.exitDoorPos = exitDoorPos

# class SimpleMultiRoomEnv(MiniGridEnv):
#     """
#     Environment with multiple rooms (subgoals)
#     """

#     def __init__(self,
#         minNumRooms,
#         maxNumRooms,
#         maxRoomSize=10,
#         seed = 5
    
#     ):
#         assert minNumRooms > 0
#         assert maxNumRooms >= minNumRooms
#         assert maxRoomSize >= 4

#         self.minNumRooms = minNumRooms
#         self.maxNumRooms = maxNumRooms
#         self.maxRoomSize = maxRoomSize

#         self.rooms = []

#         super(SimpleMultiRoomEnv, self).__init__(
#             grid_size=25,
#             max_steps=self.maxNumRooms * 20, seed = seed
#         )

#     def _gen_grid(self, width, height):
#         roomList = []

#         # Choose a random number of rooms to generate
#         #numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)
#         numRooms = self.maxNumRooms
#         while len(roomList) < numRooms:
#             curRoomList = []

#             x = self._rand_int(0, width - 2)
#             y = self._rand_int(0, width - 2)
#             print('x, y', x, y)
#             entryDoorPos = (
#                 x,
#                 y
#             )

#             # Recursively place the rooms
#             self._placeRoom(
#                 numRooms,
#                 roomList=curRoomList,
#                 minSz=4,
#                 maxSz=self.maxRoomSize,
#                 entryDoorWall=2,
#                 entryDoorPos=entryDoorPos
#             )

#             if len(curRoomList) > len(roomList):
#                 roomList = curRoomList

#         # Store the list of rooms in this environment
#         assert len(roomList) > 0
#         self.rooms = roomList

#         # Create the grid
#         self.grid = Grid(width, height)
#         wall = Wall()

#         prevDoorColor = None

#         # For each room
#         for idx, room in enumerate(roomList):

#             topX, topY = room.top
#             sizeX, sizeY = room.size

#             # Draw the top and bottom walls
#             for i in range(0, sizeX):
#                 self.grid.set(topX + i, topY, wall)
#                 self.grid.set(topX + i, topY + sizeY - 1, wall)

#             # Draw the left and right walls
#             for j in range(0, sizeY):
#                 self.grid.set(topX, topY + j, wall)
#                 self.grid.set(topX + sizeX - 1, topY + j, wall)

#             # If this isn't the first room, place the entry door
#             if idx > 0:
#                 # Pick a door color different from the previous one
#                 doorColors = set(COLOR_NAMES)
#                 if prevDoorColor:
#                     doorColors.remove(prevDoorColor)
#                 # Note: the use of sorting here guarantees determinism,
#                 # This is needed because Python's set is not deterministic
#                 doorColor = self._rand_elem(sorted(doorColors))

#                 entryDoor = Door(doorColor)
#                 self.grid.set(*room.entryDoorPos, entryDoor)
#                 prevDoorColor = doorColor

#                 prevRoom = roomList[idx-1]
#                 prevRoom.exitDoorPos = room.entryDoorPos

#         # Randomize the starting agent position and direction
#         self.place_agent(roomList[0].top, roomList[0].size)

#         # Place the final goal in the last room
#         self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

#         self.mission = 'traverse the rooms to get to the goal'

#     def _placeRoom(
#         self,
#         numLeft,
#         roomList,
#         minSz,
#         maxSz,
#         entryDoorWall,
#         entryDoorPos
#     ):
#         # Choose the room size randomly
#         sizeX = self._rand_int(minSz, maxSz+1)
#         sizeY = self._rand_int(minSz, maxSz+1)

#         # The first room will be at the door position
#         if len(roomList) == 0:
#             topX, topY = entryDoorPos
#         # Entry on the right
#         elif entryDoorWall == 0:
#             topX = entryDoorPos[0] - sizeX + 1
#             y = entryDoorPos[1]
#             topY = self._rand_int(y - sizeY + 2, y)
#         # Entry wall on the south
#         elif entryDoorWall == 1:
#             x = entryDoorPos[0]
#             topX = self._rand_int(x - sizeX + 2, x)
#             topY = entryDoorPos[1] - sizeY + 1
#         # Entry wall on the left
#         elif entryDoorWall == 2:
#             topX = entryDoorPos[0]
#             y = entryDoorPos[1]
#             topY = self._rand_int(y - sizeY + 2, y)
#         # Entry wall on the top
#         elif entryDoorWall == 3:
#             x = entryDoorPos[0]
#             topX = self._rand_int(x - sizeX + 2, x)
#             topY = entryDoorPos[1]
#         else:
#             assert False, entryDoorWall

#         # If the room is out of the grid, can't place a room here
#         if topX < 0 or topY < 0:
#             return False
#         if topX + sizeX > self.width or topY + sizeY >= self.height:
#             return False

#         # If the room intersects with previous rooms, can't place it here
#         for room in roomList[:-1]:
#             nonOverlap = \
#                 topX + sizeX < room.top[0] or \
#                 room.top[0] + room.size[0] <= topX or \
#                 topY + sizeY < room.top[1] or \
#                 room.top[1] + room.size[1] <= topY

#             if not nonOverlap:
#                 return False

#         # Add this room to the list
#         roomList.append(SimpleRoom(
#             (topX, topY),
#             (sizeX, sizeY),
#             entryDoorPos,
#             None
#         ))

#         # If this was the last room, stop
#         if numLeft == 1:
#             return True

#         # Try placing the next room
#         for i in range(0, 8):

#             # Pick which wall to place the out door on
#             wallSet = set((0, 1, 2, 3))
#             wallSet.remove(entryDoorWall)
#             exitDoorWall = self._rand_elem(sorted(wallSet))
#             nextEntryWall = (exitDoorWall + 2) % 4

#             # Pick the exit door position
#             # Exit on right wall
#             if exitDoorWall == 0:
#                 exitDoorPos = (
#                     topX + sizeX - 1,
#                     topY + self._rand_int(1, sizeY - 1)
#                 )
#             # Exit on south wall
#             elif exitDoorWall == 1:
#                 exitDoorPos = (
#                     topX + self._rand_int(1, sizeX - 1),
#                     topY + sizeY - 1
#                 )
#             # Exit on left wall
#             elif exitDoorWall == 2:
#                 exitDoorPos = (
#                     topX,
#                     topY + self._rand_int(1, sizeY - 1)
#                 )
#             # Exit on north wall
#             elif exitDoorWall == 3:
#                 exitDoorPos = (
#                     topX + self._rand_int(1, sizeX - 1),
#                     topY
#                 )
#             else:
#                 assert False

#             # Recursively create the other rooms
#             success = self._placeRoom(
#                 numLeft - 1,
#                 roomList=roomList,
#                 minSz=minSz,
#                 maxSz=maxSz,
#                 entryDoorWall=nextEntryWall,
#                 entryDoorPos=exitDoorPos
#             )

#             if success:
#                 break

#         return True

class SimpleMultiRoomEnv(MiniGridEnv):
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
        super().__init__(width=12,
        height=17, max_steps=100)

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
        width = 12
        height = 17
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        #self.grid.horz_wall(0, 0) #y is the fixed column 
        #self.grid.horz_wall(0, height - 1)
        #self.grid.vert_wall(0, 0)
        #self.grid.vert_wall(width - 1, 0)
        outer_walls = []
        # for i in range(0,9):
        #     outer_walls.append((0, i))
        #     outer_walls.append((8, i))
        # for i in range(1,8):
        #     outer_walls.append((i, 0))
        #     outer_walls.append((i, 8))
        self.blocked_states=  [(0,3), (2,3), (3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (9,3), (10,3), (3,0), (3,1), (3,2), (7,0), (7,1), (7,2), (3,4), (3,5), (7,4), (7,6), (7,7), (3,7), (4,7), (5,7), (6,7), (8,7), (9,7), (10,7), (0,6), (1,6), (6,8), (6,9), (6,10), (0,10), (1,10), (2,10), (3,10),(8,10), (9,10),(10,10), (5,12), (5,13), (5,14), (5,15), (6,12), (7,12), (8,13), (8,14), (8,15)]


        for state in self.blocked_states:
            state_x = state[0]+1
            state_y = state[1]+1
            print(state_x, state_y)
            self.grid.place_wall(state_x, state_y)

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

# class SimpleMultiRoomEnv(MultiRoomEnv):
#     def __init__(self):
#         super().__init__(
#             minNumRooms=6,
#             maxNumRooms=2,
#             maxRoomSize=4
#         )

# class MultiRoomEnvN4S5(MultiRoomEnv):
#     def __init__(self):
#         super().__init__(
#             minNumRooms=4,
#             maxNumRooms=4,
#             maxRoomSize=5
#         )

# class SimpleMultiRoom_v0(SimpleMultiRoomEnv):
#     def __init__(self):
#         super().__init__(
#             minNumRooms=6,
#             maxNumRooms=6
#         )
class SimpleMultiRoomEnv0(SimpleMultiRoomEnv):
    def __init__(self):
        agent_pos = np.array([1,1])
        goal_pos = np.array([5,6])
        super().__init__(agent_pos=agent_pos, goal_pos=goal_pos)
# register(
#     id='MiniGrid-MultiRoom-N2-S4-v0',
#     entry_point='gym_minigrid.envs:MultiRoomEnvN2S4'
# )

# register(
#     id='MiniGrid-MultiRoom-N4-S5-v0',
#     entry_point='gym_minigrid.envs:MultiRoomEnvN4S5'
# )

register(
    id='MiniGrid-Simple-MultiRoom-v0',
    entry_point='gym_minigrid.envs.simplemultiroom:SimpleMultiRoomEnv0'
)
