#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import viterbi_comparison

# OPTIONS
topology = "fully-connected" # not used yet
n_states = 2
eta = 1
n_tests = 1
verbose = True
f_eval_prob = "results/evaluation_prob"
f_compr_eval_prob = "results/compressed_evaluation_prob"
f_decod_prob = "results/decoding_prob"
f_compr_decod_prob = "results/compressed_decoding_prob"
# SCRIPTS PATHS
f_generate_states = "python/generate_states.py"
f_generate_model = "python/create_model_file.py"
f_generate_data = "python/generate_data.py"
f_wahmm = "bin/WaHMM"
# MAIN PROGRAM ARGUMENTS
# --import data/model --obs data/bin_observations --binary --tofile
# --evaluation --decoding --training
wahmm_args = []
wahmm_args.append("bin/WaHMM")
wahmm_args.append("--import")
wahmm_args.append("data/model")
wahmm_args.append("--obs")
wahmm_args.append("data/bin_observations")
wahmm_args.append("--binary")
wahmm_args.append("--tofile")
eval_std_args = wahmm_args.copy()
eval_std_args.append("--evaluation")
decod_std_args = wahmm_args.copy()
decod_std_args.append("--decoding")
train_std_args = wahmm_args.copy()
train_std_args.append("--training")
eval_compr_args = eval_std_args.copy()
eval_compr_args.append("--compressed")
decod_compr_args = decod_std_args.copy()
decod_compr_args.append("--compressed")
train_compr_args = train_std_args.copy()
train_compr_args.append("--compressed")


if verbose:
    print("[Test] Generating states... ",end="",flush=True)
arguments = [f_generate_states, str(eta), str(n_states)]
subprocess.call(arguments)
if verbose:
    print("done.",flush=True)

if verbose:
    print("[Test] Generating model... ",end="",flush=True)
subprocess.call(f_generate_model)
if verbose:
    print("done.",flush=True)

evaluation_errors = []
decoding_errors = []
decoding_paths_std_errors = []
decoding_paths_compr_errors = []
for iteration in range(0, n_tests):
    # Step 1: data generation
    if verbose:
        print("[Test] Generating data... ",end="",flush=True)
    # subprocess.call(f_generate_data)
    if verbose:
        print("done.",flush=True)

    # Step 2: evaluation problem
    if verbose:
        print("[Test] Running WaHMM uncompressed evaluation...")
    subprocess.call(eval_std_args)
    if verbose:
        print("[Test] WaHMM uncompressed evaluation finished.")
    if verbose:
        print("[Test] Running WaHMM compressed evaluation...")
    subprocess.call(eval_compr_args)
    if verbose:
        print("[Test] WaHMM compressed evaluation finished.")

    in_eval_file = open(f_eval_prob, "r")
    evaluation_prob = float(in_eval_file.read())
    in_eval_file.close()
    in_eval_file = open(f_compr_eval_prob, "r")
    compressed_evaluation_prob = float(in_eval_file.read())
    in_eval_file.close()
    if verbose:
        print("[Test] Uncompressed evaluation probability:", evaluation_prob)
        print("[Test] Compressed evaluation probability:", compressed_evaluation_prob)
        print("[Test] Error:", evaluation_prob - compressed_evaluation_prob)
    evaluation_errors.append(evaluation_prob - compressed_evaluation_prob)

    # Step 3: decoding problem
    if verbose:
        print("[Test] Running WaHMM uncompressed decoding...")
    subprocess.call(decod_std_args)
    if verbose:
        print("[Test] WaHMM uncompressed decoding finished.")
    if verbose:
        print("[Test] Running WaHMM compressed decoding...")
    subprocess.call(decod_compr_args)
    if verbose:
        print("[Test] WaHMM compressed decoding finished.")

    in_decod_file = open(f_decod_prob, "r")
    decoding_prob = float(in_decod_file.read())
    in_decod_file.close()
    in_decod_file = open(f_compr_decod_prob, "r")
    compressed_decoding_prob = float(in_decod_file.read())
    in_decod_file.close()
    if verbose:
        print("[Test] Uncompressed decoding probability:", decoding_prob)
        print("[Test] Compressed decoding probability:", compressed_decoding_prob)
        print("[Test] Error:", decoding_prob - compressed_decoding_prob)
    decoding_errors.append(decoding_prob - compressed_decoding_prob)
    path_errors = viterbi_comparison.count_differences_uncompressed()
    if verbose:
        print("[Test] Errors in path for uncompressed:", path_errors)
    decoding_paths_std_errors.append(path_errors)
    path_errors = viterbi_comparison.count_differences_compressed()
    if verbose:
        print("[Test] Errors in path for compressed:", path_errors)
    decoding_paths_compr_errors.append(path_errors)

    # Step 4: training problem
