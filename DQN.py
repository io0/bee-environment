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
import matplotlib
from collections import namedtuple
from datetime import datetime


matplotlib.style.use('ggplot')

RunStats = namedtuple('RunStats', ['algorithm', 'steps', 'returns'])


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

epochs = 400
losses = []
paths = []
flowers = []
stats = RunStats(
        algorithm="DQN_1",
        steps=np.zeros(epochs),
        returns=np.zeros(epochs))

for i in range(epochs):
    # state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state_ = env.reset()
    state = Variable(torch.from_numpy(state_).float())
    ret = 0
    
    #while game still in progress
    for t in itertools.count():
        
        qval = model(state)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action = env.action_space_sample()
        else:
            action = (np.argmax(qval_))
        
        next_state_, reward, done, _ = env.step(action)
        next_state_ = next_state_ + np.random.rand(1,l1)/10.0
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
        ret += reward
        if done:
            break
    stats.steps[i] = t
    stats.returns[i] = ret
    if i % 20 == 0:
        paths.append(env.log)
        flowers.append(env.flower)
    
    print(i, loss.item(), t)
    if epsilon > 0.1:
        epsilon -= (1/epochs)
def timestring():
    now = datetime.now()
    return now.strftime("%m_%d_%H_%M_%S_") 

def performance(stats):
    plt.figure()
    plt.plot(stats.steps)
    plt.xlabel("epoch")
    plt.ylabel("number of steps")
    # plt.plot(stats.returns)
def moving_average(stats):
    N=10
    moving_average = np.convolve(stats.steps, np.ones((N,))/N, mode='valid')
    plt.plot(moving_average)
    
def loss_curve(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.savefig(fname=timestring() + "loss")

performance(stats)
moving_average(stats)
loss_curve(losses)

def plot_paths(paths, flowers, idx=None):
    plt.figure()
    if idx is None:
        idx = int(len(paths)/10) - 1
    flowers = flowers[idx*10:(idx+1)*10]
    trace = paths[idx*10:(idx+1)*10]
    for path, flower in zip(trace, flowers):
        r = np.array(path).squeeze()
        plt.plot(r[:,0], r[:,1])
        plt.scatter(flower[0], flower[1])
    plt.savefig(fname=timestring() + "path_" + str(idx))
        
plot_paths(paths, flowers)
plot_paths(paths, flowers, idx=0)