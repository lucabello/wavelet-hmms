#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import subprocess
# SYNTAX: single_test.py <eta> <n_states>

if "-h" in sys.argv or "--help" in sys.argv:
    print("WaHMM Single-test python wrapper")
    print("single_test.py <eta> <n_states> <topology> <sequence_length> " \
        "<n_transitions>")
    print("<eta> is the state separation")
    print("<n_states> is the number of states")
    print("<topology> is either FC, CI or LR")
    print("<sequence_length> refers to the observations sequence")
    print("<n_transitions> is the expected number of transitions")
    exit(0)

if len(sys.argv) != 6:
    print("Error. Please execute as: single_test.py <eta> <n_states> " \
        "<topology> <sequence_length> <n_transition>")
    exit(1)

# read from command line
eta = float(sys.argv[1]) # 0 is total overlap, 1 is non-overlapping within
    # +/- 3 sigmas
n_states = int(sys.argv[2])
topology = sys.argv[3]
sequence_length = int(sys.argv[4])
n_transitions = int(sys.argv[5])

# generate model
arguments = ["python/generate_model.py", str(eta), str(n_states), \
    str(1 - (n_transitions / sequence_length) ), topology]
subprocess.call(arguments)
# generate data
arguments = ["python/generate_data.py", str(sequence_length)]
subprocess.call(arguments)
# execute algorithms
wahmm_args = []
wahmm_args.append("bin/WaHMM")
wahmm_args.append("--model")
wahmm_args.append("data/model")
wahmm_args.append("--estimate")
wahmm_args.append("data/kmeans_model")
wahmm_args.append("--obs")
wahmm_args.append("data/bin_observations")
wahmm_args.append("--binary")
# wahmm_args.append("--silence")
# wahmm_args.append("--tofile")
wahmm_args.append("--evaluation")
wahmm_args.append("--decoding")
wahmm_args.append("--training")
# wahmm_args.append("--compressed")

subprocess.call(wahmm_args)
