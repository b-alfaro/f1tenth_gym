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
    def __init__(self, config: dict = None, render_mode=None, **kwargs):
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
        print('Action ranges:')
        print(self.action_range)

        print('Observation space:')
        print(self.observation_space)

        self.total_steps = 0

        # read in csv file that contains information about potential parking spots
        track_dir = find_track_dir(self.map)

        ## for now, only doing parking along 1 wall
        parking_file = os.path.join(track_dir, "possible_targets_wall1.csv")
        self.parking_spots = np.genfromtxt(parking_file, delimiter=',')

    def _check_done(self):
        '''
        checking if rollout is done - modified from base environment so that done conditions are
        either crashes or being close to the target parking configuration in both position and 
        orientation
        
        important implementation note: the f110 gym in this repo implements yaws in the range 0 to
        2pi, not -pi to pi, so we need to remap for rewards/checking for done condition
        '''

        done = False
        pos_eps = 0.05 # allowable position error
        ori_eps = np.deg2rad(5.0) # allowable ori error

        if hasattr(self, 'goal_pos'):
            yaw = self.poses_theta[self.ego_idx]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            done = done or ((np.linalg.norm(self.sim.agent_poses[self.ego_idx, :2] - self.goal_pos, 2) < pos_eps
                            and np.abs(yaw - self.goal_ori) < ori_eps))
        
        done = done or self.collisions[self.ego_idx]
        return bool(done), False # second return needed for super's step func
    
    # reward scales
    POS_SCALE = 10.0
    ORI_SCALE = 10.0
    CRASH_SCALE = -1.0
    POSE_CURRICULUM = int(2e5)
    def _get_reward(self):
        reward = 0.0
        i = self.ego_idx
        pos_radius = 3.0 * 0.5 ** (self.total_steps // self.POSE_CURRICULUM)
        ori_radius = np.deg2rad(45.0) * 0.5 ** (self.total_steps // self.POSE_CURRICULUM)
        if hasattr(self, 'goal_pos'):
            pos_error = np.linalg.norm(self.sim.agent_poses[i, :2] - self.goal_pos, 2)
            yaw = self.poses_theta[i]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            ori_error = np.abs(yaw - self.goal_ori)
        else:
            # set to large numbers so exponent is effectively 0
            pos_error = 1e3
            ori_error = np.pi
        
        reward += self.POS_SCALE * np.exp(-(pos_error) ** 2 / pos_radius)
        if pos_error < 0.1:
            reward += self.ORI_SCALE * np.exp(-(ori_error) ** 2 / ori_radius)

        self.highest_seen_reward = max(reward, self.highest_seen_reward)
        reward /= self.highest_seen_reward
        reward += self.CRASH_SCALE * float(self.collisions[i])
        return reward

    def step(self, action):
        # remap to meaningful values
        action = action * self.action_range
        self.total_steps += 1
        obs, reward, done, truncated, info = super().step(action)
        # add in for timeout/truncation after 30 sec
        truncated = self.current_time > 30.0
        # add in helpful info stats
        if hasattr(self, 'goal_pos'):
            pos_error = self.sim.agent_poses[self.ego_idx, :2] - self.goal_pos
            info['custom/position_error'] = np.linalg.norm(pos_error, 2)
            yaw = self.poses_theta[self.ego_idx]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            ori_error = yaw - self.goal_ori
            info['custom/ori_error'] = np.abs(ori_error)

            # modify observations to be in error coordinates
            obs['pose'][:2] = pos_error
            obs['pose'][-1] = ori_error
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

    def _generate_parking(self, clearance=0.5):
        rand_idx = np.arange(self.parking_spots.shape[0])
        rand_idx = np.random.choice(rand_idx)
        rand_spot = self.parking_spots[rand_idx]
        dxs = [-clearance, clearance] # local coordinate x-offset of neighboring cars
    
        # dimensions of other cars blocking spot
        szx = 0.5
        szy = 0.3
        x, y, yaw = rand_spot # these yaws are in [-pi, pi]
        self.goal_pos = rand_spot[:2]
        self.goal_ori = yaw
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        for dx in dxs:
            # define coordinates of the spot in local coordinates
            car_pts = np.array([[dx, -szy / 2],
                                [dx, szy / 2],
                                [dx + np.sign(dx) * szx, szy / 2],
                                [dx + np.sign(dx) * szx, -szy / 2]])
            world_pts = R @ car_pts.T + np.array([[x],[y]])
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
        rand_idx = np.arange(self.parking_spots.shape[0])
        rand_idx = np.random.choice(rand_idx)
        rand_spot = self.parking_spots[rand_idx]
        x, y, yaw = rand_spot
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        local_pt = np.array([0, 0.5])
        poses = np.zeros((3,))
        poses[:2] = R @ local_pt[:2] + np.array([x, y])
        poses[-1] = yaw
        poses = np.expand_dims(poses, axis=0)

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
