# gym imports
import gymnasium as gym

# others
import numpy as np

from .action import CarAction, from_single_to_multi_action_space

# base classes
from .base_classes import DynamicModel, Simulator
from .integrator import IntegratorType
from .observation import observation_factory
from .rendering import make_renderer
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update
from f110_env import F110Env
from f1tenth_gym.envs.track.utils import find_track_dir
import os
import cv2


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
        self.action_range = np.array([self.params['s_max'], 2.0]) # capping speed at 2 m/s

        # read in csv file that contains information about potential parking spots
        track_dir = find_track_dir(self.map)
        parking_file = os.path.join(track_dir, "possible_targets.csv")
        self.parking_spots = np.genfromtxt(parking_file, delimiter=',')

    def _check_done(self):
        '''
        checking if rollout is done - modified from base environment so that done conditions are
        either crashes or being close to the target parking configuration in both position and 
        orientation
        
        important implementation note: the f110 gym in this repo implements yaws in the range 0 to
        2pi, not -pi to pi
        '''

        done = False
        pos_eps = 0.05 # allowable position error
        ori_eps = np.deg2rad(5.0) # allowable ori error
    
        if hasattr(self, 'goal_pos'):
            done = done or (np.linalg.norm(self.sim.agent_poses[self.ego_idx, :2] - self.goal_pos, 2) < pos_eps
                            and np.abs(self.poses_theta[self.ego_idx] - self.goal_ori) < ori_eps)
        
        done = done or self.collisions[self.ego_idx]
        return done, False # second return needed for super's step func
    

    
    def _get_reward(self):
        pass

    def step(self, action):
        # remap to meaningful values
        action = action * self.action_range
        return super().step(action)
    
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

    def _generate_parking(self, clearance=0.3):
        rand_idx = np.arange(self.parking_spots.shape[0])
        rand_idx = np.random.choice(rand_idx)
        rand_spot = self.parking_spots[rand_idx]
        dxs = [-clearance, clearance] # local coordinate x-offset of neighboring cars
    
        # dimensions of other cars blocking spot
        szx = 0.5
        szy = 0.3
        x, y, yaw = rand_spot
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        for dx in dxs:
            # define coordinates of the spot in local coordinates
            car_pts = np.array([[dx, -szy / 2],
                                [dx, szy / 2],
                                [dx + np.sign(dx) * szx, szy / 2],
                                [dx + np.sign(dx) * szx, -szy / 2]])
            world_pts  = R @ car_pts.T + np.array([[x],[y]])
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
        return super().reset(seed=seed, options=options)