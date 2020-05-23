#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import viterbi_comparison
import utilities_io as uio
from math import exp, log, isnan
import numpy as np
import time
import networkx as nx

# OPTIONS
topology_prefix = "FC"
states = [2, 3, 5]
etas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_transitions = 10
sequence_length = 100000
n_tests = 100

verbose = False


# helper functions
# using a relative difference
def compute_error(real, measured):
    if abs(real) > abs(measured):
        maximum = abs(real)
    else:
        maximum = abs(measured)
    if real == 0 and measured == 0:
        return 0
    return abs( (real - measured) / maximum )

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
f_generate_model = "python/generate_model.py"
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
    print("topology:",topology_prefix)
    print("eta:",etas," #states:",states," #tests:",n_tests)

test_count = 1
for eta in etas:
    print("[Test] --- Using Eta:",eta,"---")
    for n_states in states:
        print("[Test] --- Model with",n_states,"states ---")
        if verbose:
            print("[Test] Generating model... ",end="",flush=True)
        arguments = [f_generate_model, str(eta), str(n_states), \
            str(1 - (n_transitions / sequence_length)), topology_prefix]
        subprocess.call(arguments)
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
            print("[Test",test_count,"] Started.")
            # Step 1: data generation
            if verbose:
                print("[Test",test_count,"] Generating data... ",end="",
                    flush=True)
            arguments = [f_generate_data, str(sequence_length)]
            subprocess.call(arguments)
            if verbose:
                print("done.",flush=True)

            # Step 2: evaluation problem
            if verbose:
                print("[Test",test_count,"] -- Running WaHMM uncompressed "
                    "evaluation...")
            start = time.perf_counter()
            subprocess.call(eval_std_args)
            end = time.perf_counter()
            evaluation_times_std.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM uncompressed evaluation "
                    "finished.")
            if verbose:
                print("[Test",test_count,"] Running WaHMM compressed "
                    "evaluation...")
            start = time.perf_counter()
            subprocess.call(eval_compr_args)
            end = time.perf_counter()
            evaluation_times_compr.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM compressed evaluation "
                    "finished.")

            in_eval_file = open(f_eval_prob, "r")
            evaluation_prob = float(in_eval_file.read())
            in_eval_file.close()
            in_eval_file = open(f_compr_eval_prob, "r")
            compressed_evaluation_prob = float(in_eval_file.read())
            in_eval_file.close()
            eval_relative_error = compute_error(evaluation_prob,
                compressed_evaluation_prob)
            if verbose:
                print("[Test",test_count,"] Uncompressed evaluation "
                    "probability:", evaluation_prob)
                print("[Test",test_count,"] Compressed evaluation "
                    "probability:", compressed_evaluation_prob)
                print("[Test",test_count,"] Relative Error:",
                    eval_relative_error)
            evaluation_errors.append(eval_relative_error)

            # Step 3: decoding problem
            if verbose:
                print("[Test",test_count,"] -- Running WaHMM uncompressed "
                    "decoding...")
            start = time.perf_counter()
            subprocess.call(decod_std_args)
            end = time.perf_counter()
            decoding_times_std.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM uncompressed decoding "
                    "finished.")
            if verbose:
                print("[Test",test_count,"] Running WaHMM compressed "
                    "decoding...")
            start = time.perf_counter()
            subprocess.call(decod_compr_args)
            end = time.perf_counter()
            decoding_times_compr.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM compressed decoding "
                    "finished.")

            in_decod_file = open(f_decod_prob, "r")
            decoding_prob = float(in_decod_file.read())
            in_decod_file.close()
            in_decod_file = open(f_compr_decod_prob, "r")
            compressed_decoding_prob = float(in_decod_file.read())
            in_decod_file.close()
            decod_relative_error = compute_error(decoding_prob,
                compressed_decoding_prob)
            if verbose:
                print("[Test",test_count,"] Uncompressed decoding probability:",
                    decoding_prob)
                print("[Test",test_count,"] Compressed decoding probability:",
                    compressed_decoding_prob)
                print("[Test",test_count,"] Relative Error:",
                    decod_relative_error)
            decoding_errors.append(decod_relative_error)
            path_errors = viterbi_comparison.count_differences_uncompressed()/ \
                sequence_length
            if verbose:
                print("[Test",test_count,"] Fraction of errors in path for "
                    "uncompressed:", path_errors)
            decoding_paths_std_errors.append(path_errors)
            path_errors = viterbi_comparison.count_differences_compressed()/ \
                sequence_length
            if verbose:
                print("[Test",test_count,"] Fraction of errors in path for "
                    "compressed:", path_errors)
            decoding_paths_compr_errors.append(path_errors)

            # Step 4: training problem
            if verbose:
                print("[Test",test_count,"] -- Running WaHMM uncompressed "
                    "training...")
            start = time.perf_counter()
            subprocess.call(train_std_args)
            end = time.perf_counter()
            training_times_std.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM uncompressed training "
                    "finished.")
            if verbose:
                print("[Test",test_count,"] Running WaHMM compressed "
                    "training...")
            start = time.perf_counter()
            subprocess.call(train_compr_args)
            end = time.perf_counter()
            training_times_compr.append(end - start)
            if verbose:
                print("[Test",test_count,"] WaHMM compressed training "
                    "finished.")

            # r stands for "real", u for "uncompressed" and c for "compressed"
            r_nstates, r_means, r_stddevs, r_trans, r_init = uio \
                .read_model()
            u_nstates, u_means, u_stddevs, u_trans, u_init = uio \
                .read_model(f_train_mod)
            c_nstates, c_means, c_stddevs, c_trans, c_init = uio \
                .read_model(f_compr_train_mod)

            ur_diff = []
            cr_diff = []

            # Save differences
            # number of states, just to remember that
            ur_diff.append(r_nstates)
            cr_diff.append(r_nstates)
            # states
            # find max KL-divergence
            u_max = -1
            c_max = -1
            for i in range(0, r_nstates):
                for j in range(0, r_nstates):
                    u_kl = kl_divergence_gaussians(r_means[i], r_stddevs[i],
                        u_means[j], u_stddevs[j])
                    if u_kl > u_max:
                        u_max = u_kl
                    c_kl = kl_divergence_gaussians(r_means[i], r_stddevs[i],
                        c_means[j], c_stddevs[j])
                    if c_kl > c_max:
                        c_max = c_kl
            # setup nx graph
            G_ur = nx.Graph()
            G_cr = nx.Graph()
            for i in range(0, r_nstates):
                G_ur.add_node("r"+str(i))
                G_ur.add_node("u"+str(i))
                G_cr.add_node("r"+str(i))
                G_cr.add_node("c"+str(i))
            for i in range(0, r_nstates):
                for j in range(0, u_nstates):
                    G_ur.add_edge("r"+str(i), "u"+str(j), \
                        weight=(u_max-kl_divergence_gaussians(r_means[i], \
                        r_stddevs[i], u_means[j], u_stddevs[j])))
                    G_cr.add_edge("r"+str(i), "c"+str(j), \
                        weight=(c_max-kl_divergence_gaussians(r_means[i], \
                        r_stddevs[i], c_means[j], c_stddevs[j])))
            ur_mates = dict(nx.max_weight_matching(G_ur, maxcardinality=True))
            cr_mates = dict(nx.max_weight_matching(G_cr, maxcardinality=True))
            for i in range(0, r_nstates):
                ur_diff.append(u_max - \
                        G_ur["r"+str(i)][ur_mates["r"+str(i)]]['weight'])
                cr_diff.append(c_max - \
                        G_cr["r"+str(i)][cr_mates["r"+str(i)]]['weight'])
            u_neworder = []
            c_neworder = []
            for i in range(0, r_nstates):
                u_neworder.append(int(ur_mates["r"+str(i)][1:]))
                c_neworder.append(int(cr_mates["r"+str(i)][1:]))
            # transitions
            for i in range(0, r_nstates):
                for j in range(0, r_nstates):
                    ur_diff.append(compute_error(r_trans[i*r_nstates+j],
                        u_trans[u_neworder[i*r_nstates+j]]))
                    cr_diff.append(compute_error(r_trans[i*r_nstates+j],
                        c_trans[c_neworder[i*r_nstates+j]]))
            # initial distributions
            ur_diff.append(compute_error(r_init[0], u_init[u_neworder[0]]))
            cr_diff.append(compute_error(r_init[0], c_init[c_neworder[0]]))

            ur_model_diff.append(ur_diff)
            cr_model_diff.append(cr_diff)

            test_count = test_count + 1


        # Save testing results to file
        prefix = "tests/" + topology_prefix + "_" + str(n_states) + "_" + \
            str(eta) + "_"
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


print("\n[Test] -- Testing is finished.")
