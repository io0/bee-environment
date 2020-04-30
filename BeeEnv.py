#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:55:16 2020

@author: marley
"""
import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import colorize, seeding, EzPickle
from scipy.spatial.distance import cdist


GRID = [15,15]
center = int(GRID[0]/2)
HIVE = np.array([center,center])
HIVE_RADIUS = 3
PURPLE = (0.5,0.4,0.9)
ORANGE =(1,0.7, 0.2)
RED = (1,0,0)
VIEWPORT_W = 600
VIEWPORT_H = 600

SCALE_X = VIEWPORT_W/GRID[0]
SCALE_Y = VIEWPORT_H/GRID[1]
def mdist(a, b, radius):
    # Manhattan distance
    return np.abs(a - b).sum() < radius

class BeeEnv(gym.Env, EzPickle):
    def __init__(self, 
                 n_agents = 4, 
                 max_episode_steps = 200,
                 logging=True,
                 fov=True,
                 enable_signaling=True,
                 n_frames = 3,
                 hive=HIVE):
        self.directions = np.array([
            [-1,0],  #left
            [0,1],  #up
            [1,0],  #right
            [0,-1], #down
            [0,0], #nothing
            ])
        self.viewer = None
        self.n_agents = n_agents
        self.n_movements = 5
        self.action_space = [spaces.Discrete(5) for i in range(self.n_agents)]
        if enable_signaling:
            self.action_space[0] = spaces.Discrete(10)
        self.fov = fov
        self.fov_buffer = [[0,0]]*n_frames
        self.n_frames = n_frames
        self.enable_signaling = enable_signaling
        self.logging = logging
        self.flower = np.array([0,0])
        self.hive = hive
        self.min_position = 0
        self.max_position = GRID[0] - 1
        self._max_episode_steps = max_episode_steps
        self.reset()
        
    def _place_flower(self):
        flower = np.array([np.random.randint(GRID[0]), np.random.randint(GRID[0])])
        while (mdist(self.hive, flower, 7)):    
            flower = np.array([np.random.randint(GRID[0]), np.random.randint(GRID[0])])
        self.flower = np.clip(flower, self.min_position, self.max_position)
    
    def _compute_fov(self):
        dir_ = self.positions[0] - self.flower
        # dir_ = dir_ / np.linalg.norm(dir_) # to unit vector
        dir_ = self.directions[np.argmin(cdist([dir_],self.directions),1)] 
        return dir_
    def _reset_fov_buffer(self):
        self.fov_buffer = [[0,0]]*self.n_frames
        
    def _in_hive(self, idx):
        return mdist(self.positions[idx], self.hive, HIVE_RADIUS)
    
    def _compute_state(self):
        coords = np.zeros([self.n_agents] + GRID)
        for idx, pos in enumerate(self.positions):
            coords[(idx,pos[0], pos[1])] = 1
        coords = np.reshape(coords, (self.n_agents, -1))
        state = []
        lead_hive = self._in_hive(0)
        if self.fov:
            fov = self._compute_fov()[0]
            if lead_hive:
                self.fov_buffer.append(fov)
                self.fov_buffer.pop(0)
            else:
                self._reset_fov_buffer()
            if self.enable_signaling:
                signaling = 0
                if self.is_signaling:
                    signaling = 1
                state.append(np.hstack((coords[0],fov, [signaling])))
            else:
                state.append(np.hstack((coords[0],fov)))
        else:
            state.append(coords[0])
        for idx in range(1,self.n_agents):
            if self.enable_signaling:
                if lead_hive and self.is_signaling and self._in_hive(idx):
                    self.communication_vec[idx] = np.array(self.fov_buffer).flatten()
                vec = np.hstack((coords[idx], self.communication_vec[idx]))
            else:
                vec = coords[idx]
            state.append(vec)
        if self.n_agents == 1:
            state = np.array(state)
        self.state = state
        
    def reset(self):
        self.positions = np.array([HIVE for i in range(self.n_agents)])
        self.pollen = np.array([False] * self.n_agents)
        self.n_pollen = 0
        self.communication_vec = np.zeros((self.n_agents, self.n_frames * 2))
        self.is_signaling = False
        self.t = 0
        if self.fov:
            self._place_flower()
        self._compute_state()
        
        if self.logging:
            # reset log
            self.log = [self.positions.copy()]
        return self.state
    
    def action_space_sample(self):
        return np.array([agent_action_space.sample() for agent_action_space in self.action_space])
    
    def step(self, action_):
        # if (not np.isin(action, np.arange(8))):
        #     print('Error: The argument A must be an integer from 0-7, indicating which action was selected.')
        action = np.array(action_)
        self.positions += self.directions[np.mod(action, self.n_movements)]
        self.positions = np.clip(self.positions, self.min_position, self.max_position)
        self.is_signaling = action[0] > self.n_movements #np.where(action > self.n_movements)
        self.pollen[(self.positions[:, 0] == self.flower[0]) & (self.positions[:, 1] == self.flower[1])] = True
        done = self.pollen.all() or self.t == self._max_episode_steps
        reward = (len(np.where(self.pollen)) - self.n_pollen) * 10 -1
        self.n_pollen = len(np.where(self.pollen))
        # if done:
        #     reward = len(np.where(self.pollen)) * 10 
        if self.logging:
            self.log.append(self.positions.copy())
        self.t += 1
        self._compute_state()
        
        return self.state, reward, done, {}
    
    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        for bee in self.positions:
            t = rendering.Transform(translation=bee*SCALE_X)
            self.viewer.draw_circle(5,30,color=PURPLE).add_attr(t)
        # draw the flower
        t = rendering.Transform(translation=self.flower*SCALE_X)
        self.viewer.draw_circle(5,30,color=RED).add_attr(t)
        t = rendering.Transform(translation=np.array(HIVE)*SCALE_X)
        self.viewer.draw_circle(30,30,color=ORANGE).add_attr(t)
        return self.viewer.render(return_rgb_array = False)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
