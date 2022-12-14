import numpy as np
import mujoco_py
from gym_fetch_RT.gym_fetch.envs import rotations, robot_env, utils
#print('in this fetch env!')
debug = False

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all MuJoCo Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, terminate_success, terminate_fail, target_offset, outer, two_dim
    ):
        """Initializes a new Fetch environment.outer

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('dense' or 'sparse' or 'very_sparse'): the reward type, i.e. dense, sparse, or very spare
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range #object is the block
        self.target_range = target_range #target is the red dot
        
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.terminate_success = terminate_success
        self.terminate_fail = terminate_fail
        self.outer = outer
        self.two_dim = two_dim

        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]
        #print('debug', debug)
        #print('doing cool stuff')
        #print("target range", self.target_range)
        #print(f'obj range', self.obj_range)
        if debug:
            print("target range", self.target_range)
            print('self.target_in_the_air', self.target_in_the_air)

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        for k, v in self.sim.model._sensor_name2id.items():
            self._touch_sensor_id_site_id.append((v, self.sim.model._site_name2id[k]))
            self._touch_sensor_id.append(v)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'very_sparse':
            if d < self.distance_threshold:
                return 1.
            else:
                return 0.
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        #print('getting obs', self.target_in_the_air)
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        #print('at the eod ')
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _get_other_obs(self):
        # RGB-D
        im0, d0 = self.sim.render(width=500, height=500, camera_name='external_camera_0', depth=True)
        im1, d1 = self.sim.render(width=500, height=500, camera_name='external_camera_1', depth=True)
        im2, d2 = self.sim.render(width=500, height=500, camera_name='external_camera_2', depth=True)

        # touch sensor data
        contact_data = self.sim.data.sensordata[self._touch_sensor_id]

        # removing the red target
        # TODO: Move this to render callback
        name = 'target0'
        target_geom_ids = [self.sim.model.geom_name2id(name)
                           for name in self.sim.model.geom_names if name.startswith('target')]
        target_mat_ids = [self.sim.model.geom_matid[gid] for gid in target_geom_ids]
        target_site_ids = [self.sim.model.site_name2id(name)
                           for name in self.sim.model.site_names if name.startswith('target')]

        self.sim.model.mat_rgba[target_mat_ids, -1] = 0
        self.sim.model.geom_rgba[target_geom_ids, -1] = 0
        self.sim.model.site_rgba[target_site_ids, -1] = 0

        return {
            'image0': im0[::-1, :, :].copy(),
            'image1': im1[::-1, :, :].copy(),
            'image2': im2[::-1, :, :].copy(),
            'depth1': d1[::-1].copy(),
            'depth2': d2[::-1].copy(),
            'contact': sensor_data.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.


    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        
        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            if self.obj_range == .15 and self.target_range== .15:
                stop = .1
            else:
                stop =self.obj_range-.01
            #print(f'object_xpos = {object_xpos} self.initial_gripper_xpos[:2] = {self.initial_gripper_xpos[:2]}')
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < stop: #0.1: # CHANGE HERE
            #this used to be indented with the above while loop
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            #print(f'object xpos= {object_xpos}')
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def get_goal(self):
        #print('self.two dim', self.two_dim)
        if self.two_dim:
            goal = self.np_random.uniform(-self.target_range, self.target_range, size=2)

            if self.target_range == .03 or self.target_range == .01:
                return goal

            else:
                while(True):
                    if (goal[0] > (-self.target_range + .04) and goal[0] < (self.target_range - .04)) or (goal[1] > (-self.target_range + .04) and goal[1] < (self.target_range - .04)):
                        goal = self.np_random.uniform(-self.target_range, self.target_range, size=2)
                        #print('goal', goal)
                    else:
                        #print((goal[0] > (-self.target_range + .04) and goal[0] < (self.target_range - .04)))
                        #print((goal[1] > (-self.target_range + .04) and goal[1] < (self.target_range - .04)))
                        #print('chosen goal', goal)
                        return goal
        else:
            #print('inside ')
            goal = self.np_random.uniform(-self.target_range, self.target_range, size=3)
            #print('self.target range', self.target_range)
            #print('og goal', goal)
            if self.target_range == .04:
                return goal

            else:
                while(True):
                    if (goal[0] > (-self.target_range + .03) and goal[0] < (self.target_range - .03)) or (goal[1] > (-self.target_range + .03) and goal[1] < (self.target_range - .03)) or  (goal[2] > (-self.target_range + .03) and goal[2] < (self.target_range - .03)):
                        goal =  self.np_random.uniform(-self.target_range, self.target_range, size=3)
                        #print('goal', goal)
                    else:
                        #print((goal[0] > (-self.target_range + .04) and goal[0] < (self.target_range - .04)))
                        #print((goal[2] > (-self.target_range + .03) and goal[2] < (self.target_range - .03)))
                        #print('chosen goal', goal)
                        return goal    
    def _sample_goal(self):
        #print('im here for some reason')
       # print('self.has_object',self.has_object)
       
        if self.has_object:
            #print('should be here sampling goals')
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
            print('self.target_in_the_air', self.target_in_the_air)
            if debug:
                print('inital gripper', self.initial_gripper_xpos[:3])
            #print('goal', goal)
        else:
            #print('self.initial_gripper_xpos[:3]', self.initial_gripper_xpos[:3])
            #self.initial_gripper_xpos[0] = 1
            if self.two_dim:
                #print('not here')
                self.initial_gripper_xpos[2] = 0.42469975

                if self.outer:
                    goal = self.initial_gripper_xpos[:2] + self.get_goal()#self.np_random.uniform(-self.target_range, self.target_range, size=2)
                else:
                    #print('geting this kind of goal')
                    goal = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
                goal = list(goal)
                goal.append(self.initial_gripper_xpos[2])
                goal = np.array(goal)
                if debug:
                    print('here')
                #print('goal', goal)
            else:
                #print('inside the 3d part')
                if self.outer:
                    #print('inside self.outer')
                    goal =  self.get_goal() + self.initial_gripper_xpos[:3] 
                    #print('should be here')
                    #print('goal that is returned', goal)
                    #print('inital gripper', self.initial_gripper_xpos[:3])
                else:
                    #print('am I here')
                    goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-.15, .15, size=3)
                    # self.initial_gripper_xpos[2] = 0.42469975
                    # goal = self.initial_gripper_xpos[:2] + self.np_random.uniform(-.15, .15, size=2)
                    # z_axis = self.initial_gripper_xpos[2] + self.np_random.uniform(0, .13, size=1)
                    # goal = list(goal) + list(z_axis)
                    # #goal.append(z_axis)
                    # goal = np.array(goal)
            #print('self.initial_gripper_xpos[:3]', self.initial_gripper_xpos[:3])
            #print('goal', goal)

            #print('using goal', goal.copy())
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        if debug:
            if (d < self.distance_threshold).astype(np.float32):
                print('success')

        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
