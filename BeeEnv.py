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

class BeeEnv(gym.Env, EzPickle):
    def __init__(self, 
                 n_agents = 4, 
                 max_episode_steps = 200,
                 logging=True,
                 fov=True,
                 signaling=True,
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
        self.action_space = [spaces.Discrete(5) for i in range(self.n_agents)]
        self.fov = fov
        self.fov_buffer = [[0,0]]*3
        self.enable_signaling = signaling
        self.logging = logging
        self.flower = np.array([0,0])
        self.hive = hive
        self.min_position = 0
        self.max_position = GRID[0] - 1
        self._max_episode_steps = max_episode_steps
        self.reset()
        
    def _place_flower(self):
        flower = np.array([np.random.randint(GRID[0]), np.random.randint(GRID[0])])
        while (np.abs(flower - self.hive).sum() < 7):    
            flower = np.array([np.random.randint(GRID[0]), np.random.randint(GRID[0])])
        self.flower = np.clip(flower, self.min_position, self.max_position)
    
    def _compute_fov(self):
        dir_ = self.positions[0] - self.flower
        # dir_ = dir_ / np.linalg.norm(dir_) # to unit vector
        dir_ = self.directions[np.argmin(cdist([dir_],self.directions),1)] 
        return dir_
    
    def _compute_state(self):
        coords = np.zeros([self.n_agents] + GRID)
        for idx, pos in enumerate(self.positions):
            coords[(idx,pos[0], pos[1])] = 1
        coords = np.reshape(coords, (self.n_agents, -1))
        state = []
        if self.fov:
            fov = self._compute_fov()[0]
            self.fov_buffer.append(fov)
            self.fov_buffer.pop(0)
            state.append(np.hstack((coords[0],fov)))
        else:
            state.append(coords[0])
        if self.n_agents > 1 and self.enable_signaling:
            for agent_coord in coords[1:]:
                vec = np.hstack((agent_coord, np.array(self.fov_buffer).flatten()))
                state.append(vec)
        if self.n_agents == 1:
            state = np.array(state)
        self.state = state
        
    def reset(self):
        self.positions = np.array([HIVE for i in range(self.n_agents)])
        self.pollen = np.array([False] * self.n_agents)
        self.t = 0
        if self.fov:
            self._place_flower()
        self._compute_state()
        
        if self.logging:
            self.log = []
        return self.state
    
    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]
    
    def step(self, action):
        # if (not np.isin(action, np.arange(8))):
        #     print('Error: The argument A must be an integer from 0-7, indicating which action was selected.')
        self.positions += self.directions[action]
        self.positions = np.clip(self.positions, self.min_position, self.max_position)

        self.pollen[(self.positions[:, 0] == self.flower[0]) & (self.positions[:, 1] == self.flower[1])] = True
        done = self.pollen.all() or self.t == self._max_episode_steps
        reward = -1
        if done:
            reward = len(np.where(self.pollen)) * 10 
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
