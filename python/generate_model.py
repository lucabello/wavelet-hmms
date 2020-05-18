#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import utilities_io as uio
# SYNTAX: generate_model.py <eta> <n_states> <topology>

# Generate the states means given a certain Eta as defined in the thesis
# standard deviation is always 1

if len(sys.argv) != 5:
    print("Error. Please execute as: generate_model.py <eta> <n_states> " \
    "<self_trans_prob> <topology>")
    print("<self_trans_prob> in standard space (e.g. 0.9 for 90%)")
    print("<topology> in 2-letters format (FC, CI or LR)")
    exit(1)

eta = float(sys.argv[1]) # 0 is total overlap, 1 is non-overlapping within
    # +/- 3 sigmas
n_states = int(sys.argv[2])
self_trans_prob = float(sys.argv[3])
topology = sys.argv[4]

means = []
#print("Using eta =",eta,"and n_states =",n_states)
# the first state is always the standard normal Z(0,1)
means.append(0.0)
for i in range(1, n_states):
    means.append(6*eta + means[i-1])
#print("Generated states:",means)

# save to file for test automation
out_file = open(uio.states_file, "w")
for i in range(0, n_states):
    out_file.write(str(means[i]))
    if i < n_states-1:
        out_file.write(" ")
out_file.close()

uio.create_model_file(self_trans_prob, topology)
