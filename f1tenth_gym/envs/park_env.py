# gym imports
import gymnasium as gym

# others
import numpy as np
import os
import cv2

# base classes
from f1tenth_gym.envs.rendering import make_renderer
from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.track.utils import find_track_dir


class ParkEnv(F110Env):
    def __init__(self, config: dict = None, render_mode=None, stage=1, **kwargs):
        super().__init__(config=config, render_mode=render_mode, **kwargs)
        
        # modify action space to be in range (-1, 1)
        self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,2),
                dtype=np.float32,
        )
        self.action_range = np.array([[self.params['s_max'], 2.0]]) # capping speed at 2 m/s
        self.highest_seen_reward = 0
        self.stage = stage  # Add stage parameter
        
        # Stage-specific parameters
        self.stage_params = {
            1: {  # Stage 1: fixed position, wide gap
                'clearance': 1.0,
                'fixed_spot': True,
                # Reward weights
                'alpha_d': -10.0,    # penalize distance
                'alpha_theta': -2.0,    # penalize misalignment
                'alpha_v': -0.5,        # penalize moving fast when near target
                'r_collision': -10.0,   # penalize collisions
                'r_success': +100.0,    # reward success
                'r_step_penalty': -0.05  # penalize time
            },
            2: {  # Stage 2: fixed position, standard gap
                'clearance': 0.5,
                'fixed_spot': True,
                # Reward weights
                'alpha_d': -4.0,       
                'alpha_theta': -1.5,   
                'alpha_v': -0.4,       
                'r_collision': -15.0,  
                'r_success': +100.0,   
                'r_step_penalty': -0.08
            },
            3: {  # Stage 3: random positions
                'clearance': 0.5,
                'fixed_spot': False,
                # Reward weights
                'alpha_d': -5.0,      
                'alpha_theta': -2.0,  
                'alpha_v': -0.5,      
                'r_collision': -20.0, 
                'r_success': +100.0,  
                'r_step_penalty': -0.1
            }
        }
        
        print('Action ranges:')
        print(self.action_range)

        print('Observation space:')
        print(self.observation_space)

        self.total_steps = 0
        self.waypoint_pos = np.zeros((3,2))
        self.waypoint_ori = np.zeros((3,))
        self.waypoint_idx = 0
        self.start_pose = np.zeros((1,3))

        # read in csv file that contains information about potential parking spots
        track_dir = find_track_dir(self.map)
        parking_file = os.path.join(track_dir, "possible_targets_wall1.csv")
        self.parking_spots = np.genfromtxt(parking_file, delimiter=',')

    def _check_done(self):
        '''
        checking if rollout is done - modified from base environment so that done conditions are
        either crashes or being close to the target parking configuration in both position and 
        orientation
        '''
        done = False
        pos_eps = 0.1  # allowable position error
        ori_eps = np.deg2rad(5.0)  # allowable ori error

        if hasattr(self, 'waypoint_pos'):
            yaw = self.poses_theta[self.ego_idx]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            curr_pos = self.sim.agent_poses[self.ego_idx, :2]
            goal_pos = self.waypoint_pos[0]  # Only one waypoint now
            goal_ori = self.waypoint_ori[0]  # Only one orientation target
            
            # Check if we've reached the target
            done = (np.linalg.norm(curr_pos - goal_pos, 2) < pos_eps and 
                   np.abs(yaw - goal_ori) < ori_eps)
        
        done = done or self.collisions[self.ego_idx]
        return bool(done), False  # second return needed for super's step func
    
    def _get_reward(self):
        """
        Compute reward based on distance to target, orientation error, velocity, and terminal conditions
        """
        i = self.ego_idx
        
        # Get current state
        if hasattr(self, 'waypoint_pos'):
            goal_pos = self.waypoint_pos[0]
            goal_ori = self.waypoint_ori[0]
            pos_error = np.linalg.norm(self.sim.agent_poses[i, :2] - goal_pos, 2)
            yaw = self.poses_theta[i]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            theta_err = yaw - goal_ori
        else:
            pos_error = 1e3
            theta_err = np.pi
        
        # Get velocity
        v = np.linalg.norm(self.sim.agent_poses[i, 3:5], 2)  # linear velocity magnitude
        
        # Get stage-specific weights
        params = self.stage_params[self.stage]
        alpha_d = params['alpha_d']
        alpha_theta = params['alpha_theta']
        alpha_v = params['alpha_v']
        r_collision = params['r_collision']
        r_success = params['r_success']
        r_step_penalty = params['r_step_penalty']

        # Base reward
        reward = alpha_d * pos_error + alpha_theta * abs(theta_err)

        # Encourage stopping near the goal
        if pos_error < 0.5:
            reward += alpha_v * abs(v)

        # Constant time penalty
        reward += r_step_penalty

        # Terminal rewards
        if self.collisions[i]:
            reward += r_collision
            return reward, True
        elif self._check_done()[0]:  # Check if we've reached the target
            reward += r_success
            return reward, True

        return reward, False

    def _world_to_local(self, vec: np.ndarray):
        yaw = self.poses_theta[self.ego_idx]
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([[c,  s],
                        [-s, c]])
        return R @ vec

    def step(self, action):
        # remap to meaningful values
        action = action * self.action_range
        self.total_steps += 1
        obs, reward, done, truncated, info = super().step(action)
        
        # Get new reward and done flag
        reward, done = self._get_reward()
        
        # add in for timeout/truncation after 30 sec
        truncated = self.current_time > 30.0
        
        # add in helpful info stats
        if hasattr(self, 'waypoint_pos'):
            goal_pos = self.waypoint_pos[0]  # Only one waypoint now
            goal_ori = self.waypoint_ori[0]  # Only one orientation target
            pos_error = self.sim.agent_poses[self.ego_idx, :2] - goal_pos
            info['custom/position_error'] = np.linalg.norm(pos_error, 2)
            yaw = self.poses_theta[self.ego_idx]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            ori_error = yaw - goal_ori
            info['custom/ori_error'] = np.abs(ori_error)

            # modify observations to be in error coordinates
            obs['pose'][:2] = self._world_to_local(pos_error)
            obs['pose'][-1] = ori_error
            obs['waypoint_idx'] = 0  # Always 0 since we only have one waypoint
            info['custom/waypoint_idx'] = 0

        return obs, reward, done, truncated, info
    
    def _update_map_from_track(self):
        self.sim.set_map(self.track)   

    def _to_img(self, x, y):
        '''
        takes world x and y coordinates and returns them as coordinates in the occupancy grid image
        '''
        ox, oy, yaw = self.track.spec.origin
        dx = x - ox
        dy = y - oy
        c = np.cos(-yaw)
        s = np.sin(-yaw)
        x = c * dx - s * dy
        y = s * dx + c * dy
        scale = 0.05
        x /= scale
        y /= scale
        return np.column_stack((x, y)).astype(np.int32)

    def _generate_parking(self):
        params = self.stage_params[self.stage]
        if params['fixed_spot']:
            # Use first parking spot for fixed position stages
            x, y, yaw = self.parking_spots[0]
        else:
            # Random spot for stage 3
            rand_idx = np.random.choice(np.arange(self.parking_spots.shape[0]))
            x, y, yaw = self.parking_spots[rand_idx]
        
        clearance = params['clearance']
        dxs = [-clearance, clearance]  # local coordinate x-offset of neighboring cars
    
        # dimensions of other cars blocking spot
        szx = 0.75
        szy = 0.8

        # update waypoints - only use center point
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        T = np.array([[x],[y]])
        
        # Single waypoint at center of parking space
        waypoint_pos = np.array([[0.0, 0.0]])
        waypoint_pos = R @ waypoint_pos.T + T
        self.waypoint_pos = waypoint_pos.T
        self.waypoint_ori = np.array([yaw])  # Single orientation target
        self.waypoint_ori = (self.waypoint_ori + np.pi) % (2 * np.pi) - np.pi

        # Start position remains the same
        start_pos = np.array([[-clearance - szx / 2, 1.5 * szy]])
        start_pos = R @ start_pos.T + T
        self.start_pose[0, :2] = start_pos.T
        self.start_pose[0, -1] = yaw
        
        # Draw blocking cars
        for dx in dxs:
            car_pts = np.array([[dx, -szy / 2],
                              [dx, szy / 2],
                              [dx + np.sign(dx) * szx, szy / 2],
                              [dx + np.sign(dx) * szx, -szy / 2]])
            world_pts = R @ car_pts.T + T
            world_pts = world_pts.T
            ixy = self._to_img(world_pts[:, 0], world_pts[:, 1])
            cv2.drawContours(self.track.occupancy_map, [ixy], 0, (0, 0, 0), -1)
        
        self._update_map_from_track()

    def reset(self, seed=None, options=None):
        '''
        general steps:
        1) reset occupancy grid to that of the original map to despawn the parking spots from last run
        2) generate a random spot and update the simulator's map
        3) remake renderers so that we can see the parking spots at eval time
        4) initialize car at random position using super's reset func and return
        '''
        self.update_map(self.map)
        self._generate_parking()
        self.waypoint_idx = 0
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=self.render_mode,
            render_fps=self.metadata["render_fps"],
        )

        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        # self.highest_seen_reward = 0

        # states after reset
        # if options is not None and "poses" in options:
        #     poses = options["poses"]
        # else:
        #     poses = self.reset_fn.sample()

        # modified: sample another nearby parking spot (along the same wall) and have the car
        # start 0.5 m away from it - this is so that we don't have to worry about difficulties that 
        # come up due to the car spawning in a different corridor than the parking spot
        # rand_idx = np.arange(self.parking_spots.shape[0])
        # rand_idx = np.random.choice(rand_idx)
        # rand_spot = self.parking_spots[rand_idx]
        # x, y, yaw = rand_spot
        # R = np.array([[np.cos(yaw), -np.sin(yaw)],
        #             [np.sin(yaw),  np.cos(yaw)]])
        # local_pt = np.array([0, 0.5])
        # poses = np.zeros((3,))
        # poses[:2] = R @ local_pt[:2] + np.array([x, y])
        # poses[-1] = yaw
        # poses = np.expand_dims(poses, axis=0)
        
        poses = self.start_pose
        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(action)

        return obs, info
