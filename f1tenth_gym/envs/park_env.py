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

def remap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

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
        self.highest_seen_reward = -1.0
        print('Action ranges:')
        print(self.action_range)

        print('Observation space:')
        print(self.observation_space)

        self.total_steps = 0
        self.waypoint_pos = np.zeros((3,2))
        self.waypoint_ori = np.zeros((3,))
        self.waypoint_idx = 0
        self.start_pose = np.zeros((1,3))
        self.action = np.zeros((1,2))
        self.parking_mode = self.config['parking_mode']

        # read in csv file that contains information about potential parking spots
        track_dir = find_track_dir(self.map)

        ## for now, only doing parking along 1 wall
        parking_file = os.path.join(track_dir, "possible_targets.csv")
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
        pos_eps = 0.075 # allowable position error
        ori_eps = np.deg2rad(5.0) # allowable ori error

        if hasattr(self, 'waypoint_pos'):
            yaw = self.poses_theta[self.ego_idx]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            curr_pos = self.sim.agent_poses[self.ego_idx, :2]
            goal_pos = self.waypoint_pos[self.waypoint_idx]
            goal_ori = self.waypoint_ori[self.waypoint_idx]
            goal_ori = (goal_ori + np.pi) % (2 * np.pi) - np.pi
            done = done or (np.linalg.norm(curr_pos - goal_pos, 2) < pos_eps and np.abs(yaw - goal_ori) < ori_eps)
            # check if we need have reached one of the first two waypoints
            # if self.waypoint_idx < 2:          
            #     if np.linalg.norm(curr_pos - goal_pos, 2) < pos_eps and np.abs(yaw - goal_ori) < ori_eps:
            #         self.waypoint_idx += 1 
            # else:
            #     done = done or (np.linalg.norm(curr_pos - goal_pos, 2) < pos_eps and np.abs(yaw - goal_ori) < ori_eps)
        
        done = done or self.collisions[self.ego_idx]
        return bool(done), False # second return needed for super's step func
    
    # reward scales
    POS_SCALE = 0.5
    ORI_SCALE = 0.25
    CRASH_SCALE = -100.0
    STALL_PENALTY = 0.0
    POSE_CURRICULUM = int(5e5)
    FAR_PENALTY = -1.0
    def _get_reward(self):
        reward = 0.0
        i = self.ego_idx
        pos_radius = 1.0 * 0.5 ** (self.total_steps // self.POSE_CURRICULUM)
        ori_radius = np.deg2rad(40.0) * 0.5 ** (self.total_steps // self.POSE_CURRICULUM)
        if hasattr(self, 'waypoint_pos'):
            goal_pos = self.waypoint_pos[self.waypoint_idx]
            goal_ori = self.waypoint_ori[self.waypoint_idx]
            goal_pos = goal_pos - self.sim.agent_poses[i, :2]
            goal_pos = self._world_to_local(goal_pos) # convert to body coords
            pos_error = np.linalg.norm(goal_pos, 2)
            yaw = self.poses_theta[i]
            yaw = remap_angle(yaw)
            ori_error = np.abs(remap_angle(yaw - goal_ori))
            # reward for driving in the right direction
            # reward += float(np.sign(goal_pos[0]) == np.sign(self.action[0, 1]))
            
        else:
            # set to large numbers so exponent is effectively 0
            pos_error = 1e3
            ori_error = np.pi
        
        reward += self.POS_SCALE * np.exp(-(pos_error) ** 2 / pos_radius)
        reward += self.ORI_SCALE * np.exp(-(ori_error) ** 2 / ori_radius)
        if pos_error > 5.0:
            reward += self.FAR_PENALTY
        elif np.abs(self.action[0, 1]) < 0.2:
            reward += self.STALL_PENALTY


        # self.highest_seen_reward = max(reward, self.highest_seen_reward)
        # reward /= self.highest_seen_reward
        reward += self.CRASH_SCALE * float(self.collisions[i])
        return reward
    
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
        self.action = action
        self.total_steps += 1
        prev_idx = self.waypoint_idx
        obs, reward, done, truncated, info = super().step(action)
        # add a bonus to reward if we got to the next target
        reward += 10.0 * float(prev_idx != self.waypoint_idx)
        # add in for timeout/truncation after 10 sec
        truncated = self.current_time > 10.0
        reward += 200.0 * float(done and not self.collisions[self.ego_idx])
        # add in helpful info stats
        if hasattr(self, 'waypoint_pos'):
            goal_pos = self.waypoint_pos[self.waypoint_idx]
            goal_ori = self.waypoint_ori[self.waypoint_idx]
            pos_error = goal_pos - self.sim.agent_poses[self.ego_idx, :2]
            yaw = self.poses_theta[self.ego_idx]
            yaw = remap_angle(yaw)
            ori_error = remap_angle(yaw - goal_ori)

            if truncated or done:
                reward -= np.linalg.norm(pos_error) * 100.0
                self.final_pos_error = np.linalg.norm(pos_error)
                self.final_ori_error = np.abs(ori_error)

            if hasattr(self, 'final_pos_error'):
                info['custom/final_pos_error'] = self.final_pos_error
                info['custom/final_ori_error'] = self.final_ori_error

            # modify observations to be in error coordinates (modified so that pose error is now just
            # obstacle position in the body frame)
            obs['pose'][:2] = self._world_to_local(pos_error)
            obs['pose'][-1] = ori_error
            # obs['waypoint_idx'] = self.waypoint_idx
            # info['custom/waypoint_idx'] = self.waypoint_idx

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

    def _generate_parking(self, clearance=0.5, seed=None):
        rand_idx = np.arange(self.parking_spots.shape[0])
        rand_idx = np.random.choice(rand_idx)
        rand_spot = self.parking_spots[rand_idx]    
        x, y, yaw = rand_spot # these yaws are in [-pi, pi]

        # update waypoints
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        T = np.array([[x],[y]])   

        # dimensions of other cars blocking spot
        if self.parking_mode == 'parallel':
            szx = 0.75
            szy = 0.3
           
            # first waypoint: ahead to the left of the "car" in front,
            # second: corner of the car in front at a 45 deg yaw offset
            # third: the parking spot
            waypoint_pos = np.array([[clearance + szx / 2, 1.5 * szy],
                                    [clearance, 1.5 * szy],
                                    [0.0, 0.05]])
            waypoint_pos = R @ waypoint_pos.T + T
            self.waypoint_pos = waypoint_pos.T
            self.waypoint_ori = np.array([yaw, yaw + np.deg2rad(30.0), yaw])
            self.waypoint_ori = remap_angle(self.waypoint_ori)
            self.waypoint_idx = np.random.randint(0,1) + 2
            # print(self.waypoint_idx)

            if self.waypoint_idx == 0:
                start_pos = np.array([[-clearance - szx / 2, 1.5 * szy]])
                start_pos = R @ start_pos.T + T

                self.start_pose[0, :2] = start_pos.T
                self.start_pose[0, -1] = yaw
            else:
            # self.waypoint_idx = 2
                self.start_pose[0, :2] = self.waypoint_pos[self.waypoint_idx-1]
                self.start_pose[0, -1] = self.waypoint_ori[self.waypoint_idx-1]
        elif self.parking_mode == 'perpendicular':
            szx = 0.3
            szy = 0.5
            clearance = 0.25
            # only look at one waypoint - the final spot
            self.waypoint_idx = 0
            self.waypoint_pos[0] = T.squeeze()
            self.waypoint_ori[0] = remap_angle(yaw - np.pi / 2)
            rng =np.random.default_rng(seed=seed)
            start_x = rng.uniform(-2.5, -0.5)
            start_y = rng.uniform(0.5, 1.0)
            start_pos = np.array([[start_x, start_y]])
            start_pos = R @ start_pos.T + T
            start_ori = remap_angle(rng.uniform(-np.pi/6, np.pi/6) + yaw)
            start_ori = start_ori % (2 * np.pi) # since simulator is in 0 to 2 pi
            self.start_pose[0, :2] = start_pos.squeeze()
            self.start_pose[0, -1] = start_ori


        dxs = [-clearance, clearance] # local coordinate x-offset of neighboring cars
        # for dx in dxs:
        #     # define coordinates of the spot in local coordinate
        #     car_pts = np.array([[dx, -szy / 2],
        #                         [dx, szy / 2],
        #                         [dx + np.sign(dx) * szx, szy / 2],
        #                         [dx + np.sign(dx) * szx, -szy / 2]])
        #     world_pts = R @ car_pts.T + T
        #     world_pts = world_pts.T
        #     ixy = self._to_img(world_pts[:, 0], world_pts[:, 1])
        #     cv2.drawContours(self.track.occupancy_map, [ixy], 0, (0, 0, 0), -1)

        # trying to debug/get base performance: perpendicular park with no obstacles, generate one off
        # track for visualization only
        boxx = 0.1
        car_pts = np.array([[boxx, -1*szy],
                            [boxx, -2*szy],
                            [-boxx, -2*szy],
                            [-boxx, -1*szy]])
        world_pts = R @ car_pts.T + T
        world_pts = world_pts.T   
        ixy = self._to_img(world_pts[:, 0], world_pts[:, 1])
        cv2.drawContours(self.track.occupancy_map, [ixy], 0, tuple([255]*3), -1)
        
        self._update_map_from_track()


    def reset(self, seed=None, options=None):
        '''
        general steps:
        1) reset occupancy grid to that of the original map to despawn the parking spots from last run
        2) generate a random spot and update the simulator's map
        3) remake renderers so that we can see the parking spots at eval time
        4) initialize car at random position using super's reset func and return
        '''

        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        self.update_map(self.map)
        self.collisions[self.ego_idx] = 1.0
        while self.collisions[self.ego_idx] != 0.0:
            self._generate_parking(seed=seed)
            # self.waypoint_idx = 0
            self.renderer, self.render_spec = make_renderer(
                params=self.params,
                track=self.track,
                agent_ids=self.agent_ids,
                render_mode=self.render_mode,
                render_fps=self.metadata["render_fps"],
            )

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
