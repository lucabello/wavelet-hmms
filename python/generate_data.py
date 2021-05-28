#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pomegranate as pm
import numpy as np
import utilities_io as uio
import utilities_kmeans as ukm
import sys

# OPTIONS
# Choose if kmeans should be performed or not
perform_kmeans = True
# Number of model estimations to produce with kmeans
estimations_number = 10
# Write the sequence to a binary file
output_binary = True
# Write the sequence1  in a human readable format
output_readable = False
# Also produce a file for the generating path (always human readable)
output_path = True


if len(sys.argv) != 2:
    print("Error. Please execute as: generate_data.py <sequence_length>")
    exit(1)
# Lenght of the observations sequence
sequence_length = int(sys.argv[1])

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
    for count in range(0, estimations_number):
        ukm.estimate_model(k=n_states, model_file="data/kmeans_model_"+str(count), labels_file="data/kmeans_labels_"+str(count))
