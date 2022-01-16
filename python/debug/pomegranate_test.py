#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pomegranate as pm
import numpy as np
import math
import json
import utilities_io as uio
import utilities_kmeans as ukm

# OPTIONS
do_evaluation = True
do_decoding = True
do_training = True
evaluation_file = "results/pm_evaluation_prob"
viterbipath_file = "results/pm_decoding_path"
viterbilogp_file = "results/pm_decoding_prob"
training_file = "results/pm_training_model"

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

# read observations from input file
observations = uio.read_observations_binary()

# problem 1 : maximum likelihood, forward algorithm
# To compare log probabilities, you can consider the ratio between them:
# specifically, with two states we can see that:
#     prob_1/prob_2 = exp(log(prob_1) - log(prob_2))
if do_evaluation:
    print("--- PM Evaluation problem ---")
    forward_matrix = model.forward(observations)
    print(forward_matrix)
    # computer evaluation probability
    inf = float("inf")
    neg_inf = float("-inf")
    evaluation = neg_inf
    for alpha in forward_matrix[-1]:
        if evaluation == inf or alpha == inf:
            evaluation = inf
            break # the end result will be +inf no matter what
        elif evaluation == neg_inf:
            evaluation = alpha
        elif alpha == neg_inf:
            evaluation = evaluation
        elif evaluation > alpha:
            evaluation = evaluation + math.log(1 + math.exp(alpha-evaluation))
        else:
            evaluation = alpha + math.log(1 + math.exp(evaluation-alpha))
    print("P( O | lambda ):", evaluation)
    # save evaluation prob to file for future comparisons
    out_file = open(evaluation_file, "w")
    out_file.write(str(evaluation))
    out_file.close()

# problem 2 : Viterbi decoding, Viterbi algorithm
# compute the predicted_path using Viterbi algorithm and count errors
if do_decoding:
    print("--- PM Decoding problem ---")
    path_logp, path_tuples = model.viterbi(observations)
    print("P( O | Q,lambda ):",path_logp)
    state_path = []
    for tuple in path_tuples:
        state_path.append(tuple[0])
    state_path = state_path[1:]
    # save logp to file for future comparisons
    out_file = open(viterbilogp_file, "w")
    out_file.write(str(path_logp))
    out_file.close()
    # save the state path to file for future comparisons
    out_file = open(viterbipath_file, "w")
    for p in state_path:
        out_file.write(str(p) + " ")
    out_file.close()
    # print a shortened version of the state path
    print("Most likely path: ",end='')
    if len(state_path) > 10:
        for i in range(0, 5):
            print(state_path[i], end=' ')
        print("...", end=' ')
        for i in range(-5, 0):
            print(state_path[i], end=' ')
        print("")
    else:
        for i in range(0, len(state_path)):
            print(state_path[i], end=' ')
        print("")


# problem 3 : expectation maximization, baum-welch
# read model estimated with kmeans
if do_training:
    print("--- PM Training problem ---")
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
    estimate_model = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts)
    estimate_model.bake()

    estimate_model.fit(list([np.array(observations)]))
    # edges in json have the following format:
    # - (start node, end node, probability, pseudocount, label)
    print(estimate_model.to_json())

    out_file = open(training_file, "w")
    out_file.write(estimate_model.to_json())
    out_file.close()
