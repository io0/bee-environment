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


GRID = [60, 60]
HIVE = [30,30]
PURPLE = (0.5,0.4,0.9)
RED = (1,0,0)
VIEWPORT_W = 600
VIEWPORT_H = 600

SCALE_X = VIEWPORT_W/GRID[0]
SCALE_Y = VIEWPORT_H/GRID[1]

class BeeEnv(gym.Env, EzPickle):
    def __init__(self, 
                 n_agents = 4, 
                 max_episode_steps = 200,
                 logging=True):
        self.directions = np.array([
            [-1,0],  #left
            [0,1],  #up
            [1,0],  #right
            [0,-1], #down
            [0,0], #nothing
            ])
        self.viewer = None
        self.n_agents = n_agents
        self.pollen = np.array([False] * n_agents)
        self.action_space = [spaces.Discrete(5) for i in range(self.n_agents)]
        self.logging = logging
        self.flower = np.array([45,45])
        self._max_episode_steps = max_episode_steps
        self.reset()
        
    def reset(self):
        self.positions = np.array([HIVE for i in range(self.n_agents)])
        self.t = 0
        if self.logging:
            self.log = []

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]
    
    def step(self, action):
        # if (not np.isin(action, np.arange(8))):
        #     print('Error: The argument A must be an integer from 0-7, indicating which action was selected.')
        self.positions += self.directions[action]
        self.pollen[(self.positions[:, 0] == self.flower[0]) & (self.positions[:, 1] == self.flower[1])] = True
        done = self.pollen.all() or self.t == self._max_episode_steps
        reward = -1
        if done:
            reward = len(np.where(self.pollen)) * 10 
        if self.logging:
            self.log.append(self.positions.copy())
        self.t += 1
        return None, reward, done, {}
    
    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        for bee in self.positions:
            t = rendering.Transform(translation=bee*SCALE_X)
            self.viewer.draw_circle(5,30,color=PURPLE).add_attr(t)
        # draw the flower
        t = rendering.Transform(translation=self.flower*SCALE_X)
        self.viewer.draw_circle(5,30,color=RED).add_attr(t)
        return self.viewer.render(return_rgb_array = False)
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
