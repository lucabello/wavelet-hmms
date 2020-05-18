#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing
import subprocess

# Paremeters arrays
topologies = ["FC", "CI", "LR"]
states = [2, 3, 5, 10, 50, 100]
etas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Function to deploy tests
def test(params):
    print(params)
    topology = params[0]
    n_states = params[1]
    eta = params[2]
    arguments = []
    arguments.append("python/automated_test_parallel.py")
    arguments.append(topology)
    arguments.append(str(n_states))
    arguments.append(str(eta))
    subprocess.call(arguments)

# configuration
t = topologies[0]
s = states[0]
configuration = []
for eta in etas:
    configuration.append([t, s, eta])

# start tests
p = multiprocessing.Pool(10)
s = p.map(test, configuration)
