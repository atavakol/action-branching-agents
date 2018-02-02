import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib

import os

class Reacher6DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(os.path.abspath(__file__))+'/assets/reacher_6dof.xml', 2)
        self.viewer = None

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = False

        if np.linalg.norm(vec) < 0.02:
            done = True
        
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:] = [-0.1, 0, 0]
        self.viewer.cam.elevation = -60
        self.viewer.cam.distance = 1.7
        self.viewer.cam.azimuth = 0

    def reset_model(self):
        n_joints = self.model.nq - 2
        max_reachable_len = n_joints * 0.1 # .1 is the length of each link 
        min_reachable_len = 0.05 # joint ranges: inf, .8, .9, 1., 1.2, 1.4

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-max_reachable_len, high=max_reachable_len, size=2)
            
            if np.linalg.norm(self.goal) < max_reachable_len and np.linalg.norm(self.goal) > min_reachable_len: 
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = self.model.nq - 2 

        theta = self.model.data.qpos.flat[:n_joints]
        
        return np.concatenate([
            np.cos(theta),  
            np.sin(theta), 
            self.model.data.qpos.flat[n_joints:], # target position 
            self.model.data.qvel.flat[:n_joints], # joint velocities
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(visible=True, init_width=500, init_height=500, go_fast=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer