from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import glob 
from models.utils import get_model_class
import torch

class SimpleKey(Key):
    def can_overlap(self):
        """Can the agent overlap with this?"""
        return True

class SimpleGridEnv(MiniGridEnv):
    """
        Simplified mini grid environment with only movement actions
    """
    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=1000,
        see_through_walls=False,
        seed=2,
        agent_view_size=7, gamma = 0.99, sparse = False
    ):
        
        self.gamma = gamma
        self.sparse = sparse
        super().__init__(grid_size, width, height, max_steps, see_through_walls, seed, agent_view_size)
        self.action_space = spaces.Discrete(3)
        
    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
    
    
    def step(self, action):
        '''
        Always try to toggle, pick up
        '''
        self.step_count += 1
        if self.sparse:
            reward = 0
        else:
            reward = -1
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                if self.sparse:
                    reward = self.gamma**self.step_count
                else:
                    reward = 0
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        else:
            assert False, "unknown action"

        # Updated the position in front of the agent
        fwd_pos = self.front_pos

        # Update the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Pick up an object
        # Always try
        if fwd_cell and fwd_cell.can_pickup():
            if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*fwd_pos, None)       

        # Toggle/activate an object
        # Always Try
        if fwd_cell:
            if isinstance(fwd_cell, Door) and fwd_cell.is_locked: 
                fwd_cell.toggle(self, fwd_pos)

       



        if self.step_count >= self.max_steps:
            done = True

        obs = MiniGridEnv.gen_obs()

        return obs, reward, done, {}
    def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                # highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

class SimpleDoorKey(SimpleGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    
    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    # def gen_obs(self):
        
    #     image = torch.tensor(self.render('rgb_array')).permute(2,0,1)/255.0
    #     with torch.no_grad():
    #         obs = self.encoder.encode(image.unsqueeze(0))

        
    #     return obs


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)
        goal = Goal()
        self.put_obj(goal, *self._goal_default_pos)
        goal.init_pos, goal.cur_pos = self._goal_default_pos




        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        self.agent_pos = self._agent_default_pos
        print(f'self.agent_pos', self.agent_pos)
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = self._rand_int(0, 4)  # assuming random start direction



        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class SimpleDoorKey5x5(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=5)

class SimpleDoorKey6x6(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=6)

class SimpleDoorKey8x8(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=8)

class SimpleDoorKey16x16(SimpleDoorKey):
    def __init__(self):
        super().__init__(size=16)


register(
    id='MiniGrid-Simple-DoorKey-5x5-v0',
    entry_point='gym_minigrid.simpleminigrid:SimpleDoorKey5x5'
)

register(
    id='MiniGrid-Simple-DoorKey-6x6-v0',
    entry_point='gym_minigrid.simpleminigrid:SimpleDoorKey6x6'
)

register(
    id='MiniGrid-Simple-DoorKey-8x8-v0',
    entry_point='gym_minigrid.simpleminigrid:SimpleDoorKey8x8'
)

register(
    id='MiniGrid-Simple-DoorKey-16x16-v0',
    entry_point='gym_minigrid.simpleminigrid:SimpleDoorKey16x16'
)
