#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
# SYNTAX: generate_states.py <eta> <n_states>

# Generate the states means given a certain Eta as defined in the thesis
# (currently at (2.39)); standard deviation is always one

if len(sys.argv) != 3:
    print("Error. Please execute as: generate_states.py <eta> <n_states>")
    exit(1)

# OPTIONS
eta = float(sys.argv[1]) # 0 is total overlap, 1 is non-overlapping within +/- 3 sigmas
n_states = int(sys.argv[2])
filename = "data/states"

means = []
#print("Using eta =",eta,"and n_states =",n_states)
# the first state is always the standard normal Z(0,1)
means.append(0.0)
for i in range(1, n_states):
    means.append(6*eta + means[i-1])
#print("Generated states:",means)

# save to file for test automation
out_file = open(filename, "w")
for i in range(0, n_states):
    out_file.write(str(means[i]))
    if i < n_states-1:
        out_file.write(" ")
out_file.close()
