#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import viterbi_comparison
import utilities_io as uio
from math import exp, log
import numpy as np
import time

# helper functions
# Relative change, https://en.wikipedia.org/wiki/Relative_change_and_difference
def compute_error(real, measured):
    # if real is 0 something is wrong, so crash
    return abs((real - measured) / real)

# compute KL divergence between two univariate gaussians with means m and
# standard deviation s
# = log(s1/s0) + (s0^2 + (m0-m1)^2)/(2*s1^2) - 1/2
def kl_divergence_gaussians(m0, s0, m1, s1):
    a = log(s1/s0)
    b = (s0**2 + (m0-m1)**2)/(2*(s1**2))
    return a + b - 0.5

def savetofile(suffix, list):
    n = len(list)
    f = prefix + suffix
    out_file = open(f, "w")
    for i in range(0, n):
        out_file.write(str(list[i]) + " ")
    out_file.close()

# OPTIONS
topology = "fully-connected" # not used yet
# states = [2, 3, 5, 7, 11, 13]
states = [2, 3, 5, 7]
etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cases_toskip = []
# cases_toskip.append([0.1, 11])
cases_toskip.append([0.0, 0])
n_tests = 100
sequence_length = 1000000 # used ONLY to calculate relative errors in decoding
verbose = True
f_eval_prob = "results/evaluation_prob"
f_compr_eval_prob = "results/compressed_evaluation_prob"
f_decod_prob = "results/decoding_prob"
f_compr_decod_prob = "results/compressed_decoding_prob"
f_train_mod = "results/training_model"
f_compr_train_mod = "results/compressed_training_model"
# output files
f_eval_out = "evaluation"
f_eval_time_std_out = "evaluation_std_time"
f_eval_time_compr_out = "evaluation_compr_time"
f_decod_prob_out = "decoding_prob"
f_decod_path_std_out = "decoding_std_path"
f_decod_path_compr_out = "decoding_compr_path"
f_decod_time_std_out = "decoding_std_time"
f_decod_time_compr_out = "decoding_compr_time"
f_train_std_out = "training_std"
f_train_compr_out = "training_compr"
# f_train_model_std_out = "training_model_std"
# f_train_model_compr_out = "training_model_compr"
f_train_time_std_out = "training_std_time"
f_train_time_compr_out = "training_compr_time"
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
wahmm_args.append("--silence")
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
    print("=== WaHMM AUTOMATED TESTING ===")
    print("eta:",etas," #states:",states," #tests:",n_tests)

skip_index = 0
test_count = 1
for eta in etas:
    print("[Test] --- Using Eta:",eta,"---")
    for n_states in states:
        print("[Test] --- Model with",n_states,"states ---")
        if cases_toskip[skip_index][0] == eta and cases_toskip[skip_index][1] == n_states:
            print("[Test] --- Case skipped ---")
            skip_index = skip_index + 1
            continue
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

        # TEST THE MODEL
        evaluation_errors = []
        evaluation_times_std = []
        evaluation_times_compr = []
        decoding_errors = []
        decoding_paths_std_errors = []
        decoding_paths_compr_errors = []
        decoding_times_std = []
        decoding_times_compr = []
        ur_model_diff = []
        cr_model_diff = []
        # u_model = []
        # c_model = []
        training_times_std = []
        training_times_compr = []
        for iteration in range(1, n_tests+1):
            # Step 1: data generation
            if verbose:
                print("[Test",test_count,"] Generating data... ",end="",flush=True)
            subprocess.call(f_generate_data)
            if verbose:
                print("done.",flush=True)

            # Step 2: evaluation problem
            if verbose:
                print("[Test",test_count,"] -- Running WaHMM uncompressed evaluation...")
            start = time.perf_counter()
            subprocess.call(eval_std_args)
            end = time.perf_counter()
            evaluation_times_std.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM uncompressed evaluation finished.")
            if verbose:
                print("[Test",test_count,"] Running WaHMM compressed evaluation...")
            start = time.perf_counter()
            subprocess.call(eval_compr_args)
            end = time.perf_counter()
            evaluation_times_compr.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM compressed evaluation finished.")

            in_eval_file = open(f_eval_prob, "r")
            evaluation_prob = float(in_eval_file.read())
            in_eval_file.close()
            in_eval_file = open(f_compr_eval_prob, "r")
            compressed_evaluation_prob = float(in_eval_file.read())
            in_eval_file.close()
            eval_relative_error = compute_error(evaluation_prob, compressed_evaluation_prob)
            if verbose:
                print("[Test",test_count,"] Uncompressed evaluation probability:", evaluation_prob)
                print("[Test",test_count,"] Compressed evaluation probability:", compressed_evaluation_prob)
                print("[Test",test_count,"] Relative Error:", eval_relative_error)
            evaluation_errors.append(eval_relative_error)

            # Step 3: decoding problem
            if verbose:
                print("[Test",test_count,"] -- Running WaHMM uncompressed decoding...")
            start = time.perf_counter()
            subprocess.call(decod_std_args)
            end = time.perf_counter()
            decoding_times_std.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM uncompressed decoding finished.")
            if verbose:
                print("[Test",test_count,"] Running WaHMM compressed decoding...")
            start = time.perf_counter()
            subprocess.call(decod_compr_args)
            end = time.perf_counter()
            decoding_times_compr.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM compressed decoding finished.")

            in_decod_file = open(f_decod_prob, "r")
            decoding_prob = float(in_decod_file.read())
            in_decod_file.close()
            in_decod_file = open(f_compr_decod_prob, "r")
            compressed_decoding_prob = float(in_decod_file.read())
            in_decod_file.close()
            decod_relative_error = compute_error(decoding_prob, compressed_decoding_prob)
            if verbose:
                print("[Test",test_count,"] Uncompressed decoding probability:", decoding_prob)
                print("[Test",test_count,"] Compressed decoding probability:", compressed_decoding_prob)
                print("[Test",test_count,"] Relative Error:", decod_relative_error)
            decoding_errors.append(decod_relative_error)
            path_errors = viterbi_comparison.count_differences_uncompressed()/sequence_length
            if verbose:
                print("[Test",test_count,"] Fraction of errors in path for uncompressed:", path_errors)
            decoding_paths_std_errors.append(path_errors)
            path_errors = viterbi_comparison.count_differences_compressed()/sequence_length
            if verbose:
                print("[Test",test_count,"] Fraction of errors in path for compressed:", path_errors)
            decoding_paths_compr_errors.append(path_errors)

            # Step 4: training problem
            if verbose:
                print("[Test",test_count,"] -- Running WaHMM uncompressed training...")
            start = time.perf_counter()
            subprocess.call(train_std_args)
            end = time.perf_counter()
            training_times_std.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM uncompressed training finished.")
            if verbose:
                print("[Test",test_count,"] Running WaHMM compressed training...")
            start = time.perf_counter()
            subprocess.call(train_compr_args)
            end = time.perf_counter()
            training_times_compr.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM compressed training finished.")

            # r stands for "real", u for "uncompressed" and c for "compressed"
            r_nstates, r_means, r_stddevs, r_trans, r_init = uio.read_model()
            u_nstates, u_means, u_stddevs, u_trans, u_init = uio.read_model(f_train_mod)
            c_nstates, c_means, c_stddevs, c_trans, c_init = uio.read_model(f_compr_train_mod)

            ur_diff = []
            cr_diff = []


            # CHECK IF THE STATES ARE SORTED, DEBUG, IF THEY ARE NOT SORT THEM
            # WHEN YOU COMPUTE THE ERROR
            for i in range(0, len(u_means)-1):
                if u_means[i] >= u_means[i+1] :
                    print("+++ OMG SORT THE U STATES PLZ +++")
                    tmp = input()
            for i in range(0, len(c_means)-1):
                if c_means[i] >= c_means[i+1]:
                    print("+++ OMG SORT THE C STATES PLZ +++")
                    tmp = input()



            # Save differences
            # number of states, just to remember that
            ur_diff.append(r_nstates)
            cr_diff.append(r_nstates)
            # states
            for i in range(0, r_nstates):
                ur_diff.append(kl_divergence_gaussians(r_means[i], r_stddevs[i],
                    u_means[i], u_stddevs[i]))
                cr_diff.append(kl_divergence_gaussians(r_means[i], r_stddevs[i],
                    c_means[i], c_stddevs[i]))
            # transitions
            for i in range(0, r_nstates):
                for j in range(0, r_nstates):
                    ur_diff.append(compute_error(exp(r_trans[i*r_nstates+j]), exp(u_trans[i*r_nstates+j])))
                    cr_diff.append(compute_error(exp(r_trans[i*r_nstates+j]), exp(c_trans[i*r_nstates+j])))
            # initial distributions
            ur_diff.append(compute_error(r_init[0], u_init[0]))
            cr_diff.append(compute_error(r_init[0], c_init[0]))

            ur_model_diff.append(ur_diff)
            cr_model_diff.append(cr_diff)

            # Save actual models
            # u_model_file = open(f_train_mod, "r")
            # u_model_list = u_model_file.read().split()
            # u_model.append(u_model_list)
            # u_model_file.close()
            # c_model_file = open(f_compr_train_mod, "r")
            # c_model_list = c_model_file.read().split()
            # c_model.append(c_model_list)
            # c_model_file.close()

            test_count = test_count + 1


        # Save testing results to file
        prefix = "tests/FC_" + str(n_states) + "_" + str(eta) + "_"
        print("[Test] Saving files with prefix:",prefix)
        # Evaluation
        savetofile(f_eval_out, evaluation_errors)
        savetofile(f_eval_time_std_out, evaluation_times_std)
        savetofile(f_eval_time_compr_out, evaluation_times_compr)
        # Decoding
        savetofile(f_decod_prob_out, decoding_errors)
        savetofile(f_decod_path_std_out, decoding_paths_std_errors)
        savetofile(f_decod_path_compr_out, decoding_paths_compr_errors)
        savetofile(f_decod_time_std_out, decoding_times_std)
        savetofile(f_decod_time_compr_out, decoding_times_compr)
        # Training
        out_file = open(prefix+f_train_std_out, "w")
        for i in range(0, n_tests):
            for x in range(0, len(ur_model_diff[i])):
                out_file.write(str(ur_model_diff[i][x]) + " ")
            out_file.write("\n")
        out_file.close()
        out_file = open(prefix+f_train_compr_out, "w")
        for i in range(0, n_tests):
            for x in range(0, len(cr_model_diff[i])):
                out_file.write(str(cr_model_diff[i][x]) + " ")
            out_file.write("\n")
        out_file.close()
        savetofile(f_train_time_std_out, training_times_std)
        savetofile(f_train_time_compr_out, training_times_compr)

        # out_file = open(prefix+f_train_model_std_out, "w")
        # for i in range(0, n_tests):
        #     for x in range(0, len(u_model[i])):
        #         out_file.write(str(u_model[i][x]) + " ")
        #     out_file.write("\n")
        # out_file.close()
        # out_file = open(prefix+f_train_model_compr_out, "w")
        # for i in range(0, n_tests):
        #     for x in range(0, len(c_model[i])):
        #         out_file.write(str(c_model[i][x]) + " ")
        #     out_file.write("\n")
        # out_file.close()


print("\n[Test] -- Testing is finished.")
# Print results
# ATTENTION: THESE RESULTS ARE CURRENTLY RELATED TO THE LAST TEST ONLY
# print("\n=== TESTING RESULTS ===")
# print("= Evaluation Errors =")
# print(evaluation_errors)
# print("= Decoding Errors Prob =")
# print(decoding_errors)
# print("= Decoding Errors Path Std =")
# print(decoding_paths_std_errors)
# print("= Decoding Errors Path Compressed =")
# print(decoding_paths_compr_errors)
# print("= Std Training Errors from real model =")
# print(ur_model_diff)
# print("= Compressed Training Errors from real model =")
# print(cr_model_diff)
print()
