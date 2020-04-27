#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import log
import utilities_io as uio

# Creates a model file; the input params can be inserted by hand

# The file should be written with the following format
# n_states mean std_dev ... log(transitions) ... log(initial_distribution) ...

# OPTIONS
automatic = True
states_file = "data/states"
trans_prob = 10.0/100000.0 # wanted_trans_n / sequence length = 10/10^6

out_file = open(uio.model_file, "w")

if automatic:
    in_file = open(states_file, "r")
    line = in_file.read()
    string_list = line.split()
    # number of states
    n_states = len(string_list)
    out_file.write(str(n_states) + " ")
    # states distribution (mean, std_dev)
    for i in range(0, n_states):
        out_file.write(string_list[i] + " 1 ")
    # transition matrix
    self_trans_prob = 1 - trans_prob
    out_trans_prob = trans_prob / (n_states - 1)
    for i in range(0, n_states):
        for j in range(0, n_states):
            if i == j:
                out_file.write(str(log(self_trans_prob)) + " ")
            else:
                out_file.write(str(log(out_trans_prob)) + " ")
    # initial distribution
    out_file.write(str(log(1)) + " ")
    for i in range(1, n_states):
        out_file.write("-inf ")
else:
    # number of states
    out_file.write("2 ")
    # states distributions (mean, std_dev)
    out_file.write("0 1 10 1 ")
    # transition matrix
    out_file.write(str(log(0.999)) + " " + str(log(0.001)) + " ")
    out_file.write(str(log(0.001)) + " " + str(log(0.999)) + " ")
    # initial distribution
    out_file.write(str(log(1)) + " " + "-inf ")

out_file.close()
