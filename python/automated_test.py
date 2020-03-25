#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import viterbi_comparison
import utilities_io as uio

# OPTIONS
topology = "fully-connected" # not used yet
n_states = 2
eta = 1
n_tests = 10
verbose = True
f_eval_prob = "results/evaluation_prob"
f_compr_eval_prob = "results/compressed_evaluation_prob"
f_decod_prob = "results/decoding_prob"
f_compr_decod_prob = "results/compressed_decoding_prob"
f_train_mod = "results/training_model"
f_compr_train_mod = "results/compressed_training_model"
# output files
f_eval_out = "results/tests_evaluation"
f_decod_prob_out = "results/tests_decoding_prob"
f_decod_path_std_out = "results/tests_decoding_std_path"
f_decod_path_compr_out = "results/tests_decoding_compr_path"
f_train_std_out = "results/tests_std_training"
f_train_compr_out = "results/tests_compr_training"
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
wahmm_args.append("--model")
wahmm_args.append("data/model")
wahmm_args.append("--estimate")
wahmm_args.append("data/kmeans_model")
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
ur_model_diff = []
cr_model_diff = []
for iteration in range(0, n_tests):
    # Step 1: data generation
    if verbose:
        print("[Test",iteration,"] Generating data... ",end="",flush=True)
    subprocess.call(f_generate_data)
    if verbose:
        print("done.",flush=True)

    # Step 2: evaluation problem
    if verbose:
        print("[Test",iteration,"] Running WaHMM uncompressed evaluation...")
    subprocess.call(eval_std_args)
    if verbose:
        print("[Test",iteration,"] WaHMM uncompressed evaluation finished.")
    if verbose:
        print("[Test",iteration,"] Running WaHMM compressed evaluation...")
    subprocess.call(eval_compr_args)
    if verbose:
        print("[Test",iteration,"] WaHMM compressed evaluation finished.")

    in_eval_file = open(f_eval_prob, "r")
    evaluation_prob = float(in_eval_file.read())
    in_eval_file.close()
    in_eval_file = open(f_compr_eval_prob, "r")
    compressed_evaluation_prob = float(in_eval_file.read())
    in_eval_file.close()
    if verbose:
        print("[Test",iteration,"] Uncompressed evaluation probability:", evaluation_prob)
        print("[Test",iteration,"] Compressed evaluation probability:", compressed_evaluation_prob)
        print("[Test",iteration,"] Error:", evaluation_prob - compressed_evaluation_prob)
    evaluation_errors.append(evaluation_prob - compressed_evaluation_prob)

    # Step 3: decoding problem
    if verbose:
        print("[Test",iteration,"] Running WaHMM uncompressed decoding...")
    subprocess.call(decod_std_args)
    if verbose:
        print("[Test",iteration,"] WaHMM uncompressed decoding finished.")
    if verbose:
        print("[Test",iteration,"] Running WaHMM compressed decoding...")
    subprocess.call(decod_compr_args)
    if verbose:
        print("[Test",iteration,"] WaHMM compressed decoding finished.")

    in_decod_file = open(f_decod_prob, "r")
    decoding_prob = float(in_decod_file.read())
    in_decod_file.close()
    in_decod_file = open(f_compr_decod_prob, "r")
    compressed_decoding_prob = float(in_decod_file.read())
    in_decod_file.close()
    if verbose:
        print("[Test",iteration,"] Uncompressed decoding probability:", decoding_prob)
        print("[Test",iteration,"] Compressed decoding probability:", compressed_decoding_prob)
        print("[Test",iteration,"] Error:", decoding_prob - compressed_decoding_prob)
    decoding_errors.append(decoding_prob - compressed_decoding_prob)
    path_errors = viterbi_comparison.count_differences_uncompressed()
    if verbose:
        print("[Test",iteration,"] Errors in path for uncompressed:", path_errors)
    decoding_paths_std_errors.append(path_errors)
    path_errors = viterbi_comparison.count_differences_compressed()
    if verbose:
        print("[Test",iteration,"] Errors in path for compressed:", path_errors)
    decoding_paths_compr_errors.append(path_errors)

    # Step 4: training problem
    if verbose:
        print("[Test",iteration,"] Running WaHMM uncompressed training...")
    subprocess.call(train_std_args)
    if verbose:
        print("[Test",iteration,"] WaHMM uncompressed training finished.")
    if verbose:
        print("[Test",iteration,"] Running WaHMM compressed training...")
    subprocess.call(train_compr_args)
    if verbose:
        print("[Test",iteration,"] WaHMM compressed training finished.")

    # r stands for "real", u for "uncompressed" and c for "compressed"
    r_nstates, r_means, r_stddevs, r_trans, r_init = uio.read_model()
    u_nstates, u_means, u_stddevs, u_trans, u_init = uio.read_model(f_train_mod)
    c_nstates, c_means, c_stddevs, c_trans, c_init = uio.read_model(f_compr_train_mod)

    ur_diff = []
    cr_diff = []
    # number of states, just to remember that
    ur_diff.append(r_nstates)
    cr_diff.append(r_nstates)
    # states
    for i in range(0, r_nstates):
        ur_diff.append(u_means[i] - r_means[i])
        ur_diff.append(u_stddevs[i] - r_stddevs[i])
        cr_diff.append(c_means[i] - r_means[i])
        cr_diff.append(c_stddevs[i] - r_stddevs[i])
    # transitions
    for i in range(0, r_nstates):
        for j in range(0, r_nstates):
            ur_diff.append(u_trans[i*r_nstates+j] - r_trans[i*r_nstates+j])
            cr_diff.append(c_trans[i*r_nstates+j] - r_trans[i*r_nstates+j])
    # initial distributions
    for i in range(0, r_nstates):
        ur_diff.append(u_init[i] - r_init[i])
        cr_diff.append(c_init[i] - r_init[i])

    ur_model_diff.append(ur_diff)
    cr_model_diff.append(cr_diff)


# Save testing results to file
out_file = open(f_eval_out, "w")
for i in range(0, n_tests):
    out_file.write(str(evaluation_errors[i]) + " ")
out_file.close()
out_file = open(f_decod_prob_out, "w")
for i in range(0, n_tests):
    out_file.write(str(decoding_errors[i]) + " ")
out_file.close()
out_file = open(f_decod_path_std_out, "w")
for i in range(0, n_tests):
    out_file.write(str(decoding_paths_std_errors[i]) + " ")
out_file.close()
out_file = open(f_decod_path_compr_out, "w")
for i in range(0, n_tests):
    out_file.write(str(decoding_paths_compr_errors[i]) + " ")
out_file.close()
out_file = open(f_train_std_out, "w")
for i in range(0, n_tests):
    for x in range(0, len(ur_model_diff[i])):
        out_file.write(str(ur_model_diff[i][x]) + " ")
    out_file.write("\n")
out_file.close()
out_file = open(f_train_compr_out, "w")
for i in range(0, n_tests):
    for x in range(0, len(cr_model_diff[i])):
        out_file.write(str(cr_model_diff[i][x]) + " ")
    out_file.write("\n")
out_file.close()

# Print results
print("\n=== TESTING RESULTS ===")
print("= Evaluation Errors =")
print(evaluation_errors)
print("= Decoding Errors Prob =")
print(decoding_errors)
print("= Decoding Errors Path Std =")
print(decoding_paths_std_errors)
print("= Decoding Errors path Compressed =")
print(decoding_paths_compr_errors)
print("= Std Training Errors from real model =")
print(ur_model_diff)
print("= Compressed Training Errors from real model =")
print(cr_model_diff)
print()
