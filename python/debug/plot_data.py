#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import utilities_io as uio

# OPTIONS
use_binary_file = True
input_limit = 1000

if use_binary_file:
    observations = uio.read_observations_binary()
    state_path = uio.read_path()
else:
    observations = uio.read_observations(limit=input_limit)
    state_path = uio.read_path(limit=input_limit)

n_states, means, std_devs, transitions, initial = uio.read_model()
means_path = state_path
for i in range(0, len(state_path)):
    means_path[i] = means[state_path[i]]

if input_limit > 0 and use_binary_file == False:
    x = range(1,input_limit+1)
else:
    x = range(1,len(observations)+1)

plt.scatter(x, observations, color='blue')
plt.step(x, means_path, color='red')
plt.show()
