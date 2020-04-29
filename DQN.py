#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:43:35 2020

@author: marley
"""

import numpy as np
import torch
from torch.autograd import Variable
import random
from matplotlib import pylab as plt
from BeeEnv import BeeEnv
import itertools


env = BeeEnv(n_agents=1, max_episode_steps=1000)
l1 = env.state.size
l2 = 150
l3 = 100
l4 = 5
model = torch.nn.Sequential(
 torch.nn.Linear(l1, l2),
 torch.nn.ReLU(),
 torch.nn.Linear(l2, l3),
 torch.nn.ReLU(),
 torch.nn.Linear(l3,l4)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
gamma = 0.99
epsilon = 0.3

epochs = 1000
losses = []
stats = np.zeros(epochs)

for i in range(epochs):
    # state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state_ = env.reset().reshape(1,l1)
    state = Variable(torch.from_numpy(state_).float())
    
    #while game still in progress
    for t in itertools.count():
        
        qval = model(state)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action = env.action_space_sample()
        else:
            action = (np.argmax(qval_))
        
        next_state_, reward, done, _ = env.step(action)
        next_state_ = next_state_.reshape(1,l1) + np.random.rand(1,l1)/10.0
        next_state = Variable(torch.from_numpy(next_state_).float())
        newQ = model(next_state.reshape(1,l1)).data.numpy()
        maxQ = np.max(newQ)
        y = np.zeros((1,l4))
        y[:] = qval_[:]
        if reward == -1:
            update = (reward + (gamma * maxQ))
        else:
            update = reward
        y[0][action] = update
        y = Variable(torch.from_numpy(y).float())
        loss = loss_fn(qval, y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        state = next_state
        if done:
            break
    stats[i] = t
    print(i, loss.item(), t)
    if epsilon > 0.1:
        epsilon -= (1/epochs)
        
def loss_curve(losses):
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")