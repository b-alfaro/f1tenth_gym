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
                'k_d': -20.0,    # Reduced position error penalty to allow more movement
                'k_theta': -10.0, # Increased orientation penalty to encourage proper alignment
                'k_v': -1.0,     # Increased velocity penalty to encourage careful movement
                'k_collision': -50.0,  # Increased collision penalty
                'k_success': +200.0,   # Increased success reward
                'k_step_penalty': -0.01
            },
            2: {  # Stage 2: fixed position, standard gap
                'clearance': 0.5,
                'fixed_spot': True,
                # Reward weights
                'k_d': -15.0,       
                'k_theta': -4.0,    
                'k_v': -0.4,        
                'k_collision': -15.0,   
                'k_success': +100.0,    
                'k_step_penalty': -0.02 
            },
            3: {  # Stage 3: random positions
                'clearance': 0.5,
                'fixed_spot': False,
                # Reward weights
                'k_d': -15.0,      
                'k_theta': -4.0,   
                'k_v': -0.5,       
                'k_collision': -20.0,  
                'k_success': +100.0,   
                'k_step_penalty': -0.02
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
            
            # Calculate approach angle reward
            # This encourages the car to approach the parking spot at an angle
            car_to_goal = goal_pos - self.sim.agent_poses[i, :2]
            approach_angle = np.arctan2(car_to_goal[1], car_to_goal[0])
            angle_diff = np.abs(approach_angle - yaw)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            approach_reward = -5.0 * angle_diff  # Penalize large angle differences
        else:
            pos_error = 1e3
            theta_err = np.pi
            approach_reward = 0
        
        # Get velocity
        v = np.linalg.norm(self.sim.agent_poses[i, 3:5], 2)  # linear velocity magnitude
        
        # Get stage-specific weights
        params = self.stage_params[self.stage]
        k_d = params['k_d']
        k_theta = params['k_theta']
        k_v = params['k_v']
        k_collision = params['k_collision']
        k_success = params['k_success']
        k_step_penalty = params['k_step_penalty']

        # Base reward with approach angle component
        reward = k_d * pos_error + k_theta * abs(theta_err) + approach_reward

        # Encourage stopping near the goal
        if pos_error < 0.5:
            reward += k_v * abs(v)

        # Constant time penalty
        reward += k_step_penalty

        # Terminal rewards
        if self.collisions[i]:
            reward += k_collision
            return reward, True
        elif self._check_done()[0]:  # Check if we've reached the target
            reward += k_success
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

        # Draw target position
        target_img_pos = self._to_img(waypoint_pos[0], waypoint_pos[1])[0]
        cv2.circle(self.track.occupancy_map, (target_img_pos[0], target_img_pos[1]), 2, (0, 0, 255), -1)

        # Start position remains the same
        start_pos = np.array([[-clearance - szx / 2, 1.0 * szy]])
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
