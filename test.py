import gym
import numpy as np
import AtariEnv
import BeeEnv
import matplotlib.pyplot as plt
import itertools

env = BeeEnv.BeeEnv()
env.reset()
ret = 0
for t in itertools.count():
    env.render()
    next_state, reward, done,_ = env.step(env.action_space_sample()) # take a random action
    ret += reward
    if done:
        break
agent_zero = 0
trace = np.array(env.log)[:,agent_zero]
plt.plot(trace[:,0], trace[:,1])
plt.scatter(env.flower[0], env.flower[1])
env.close()