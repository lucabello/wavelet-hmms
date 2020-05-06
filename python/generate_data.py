#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pomegranate as pm
import numpy as np
import utilities_io as uio
import utilities_kmeans as ukm

# OPTIONS
# Lenght of the observations sequence
sequence_length = 100000
# Choose if kmeans should be performed or not
perform_kmeans = True
# Write the sequence to a binary file
output_binary = True
# Write the sequence1  in a human readable format
output_readable = True
# Also produce a file for the generating path (always human readable)
output_path = True

# read model from input file
n_states, means, std_devs, transitions, initial = uio.read_model()

# create Pomegranate model
dists = []
for i in range(0, n_states):
    dists.append(pm.NormalDistribution(means[i], std_devs[i]))
trans_mat = []
for i in range(0, n_states):
    trans_mat.append([])
    for j in range(0, n_states):
        trans_mat[i].append(transitions[i*n_states + j])
trans_mat = np.array(trans_mat)
starts = np.array(initial)
model = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts)
model.bake()

# generate samples from the model
if output_path:
    samples = model.sample(length=sequence_length, path=True, random_state=None)
    observations = samples[0]
    state_path = samples[1]
else:
    samples = model.sample(length=sequence_length, path=False,
        random_state=None)
    observations = samples

if output_readable:
    uio.write_observations(observations)
if output_binary:
    uio.write_observations_binary(observations)
if output_path:
    uio.write_path(state_path)

if perform_kmeans:
    ukm.estimate_model(k=n_states)
