#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import utilities_io as uio
from math import sqrt
import sys

def print(*args):
    __builtins__.print(*("%.2e" % a if isinstance(a, float) else a
                         for a in args))


# OPTIONS
n_tests = 3 # to know # of lines in training reuslts
f_ev = "evaluation"
f_de_pr = "decoding_prob"
f_de_std_pa = "decoding_std_path"
f_de_compr_pa = "decoding_compr_path"
f_tr_std = "training_std"
f_tr_compr = "training_compr"
f_ev_std_t = "evaluation_std_time"
f_ev_compr_t = "evaluation_compr_time"
f_de_std_t = "decoding_std_time"
f_de_compr_t = "decoding_compr_time"
f_tr_std_t = "training_std_time"
f_tr_compr_t = "training_compr_time"

def aggregate_measures(element_list):
    result_avg = 0.0
    result_dev = 0.0
    for e in element_list:
        ef = float(e)
        result_avg = result_avg + ef
    result_avg = result_avg / len(element_list)
    for e in element_list:
        ef = float(e)
        result_dev = result_dev + sqrt((ef - result_avg)**2)
    result_dev = result_dev / len(element_list)
    return result_avg, result_dev

def aggregate(suffix):
    f = prefix + suffix
    in_file = open(f, "r")
    list = in_file.read().split()
    in_file.close()
    return aggregate_measures(list)

def aggregate_measures_training(filename):
    # calculate std avg
    kl_avg = 0.0
    kl_dev = 0.0
    tran_avg = 0.0
    tran_dev = 0.0
    init_avg = 0.0
    init_dev = 0.0
    in_file = open(filename, "r")
    tr_list = in_file.read().split()
    index = 0
    for t in range(0, n_tests):
        n_states = int(tr_list[index])
        index = index + 1
        for i in range(0, n_states):
            kl_avg = kl_avg + float(tr_list[index])
            index = index + 1
        for i in range(0, n_states):
            for j in range(0, n_states):
                tran_avg = tran_avg + float(tr_list[index])
                index = index + 1
        init_avg = init_avg + float(tr_list[index])
        index = index + 1
    kl_avg = kl_avg / n_tests
    tran_avg = tran_avg / n_tests
    init_avg = init_avg / n_tests
    in_file.close()
    # calculate std dev
    in_file = open(filename, "r")
    tr_list = in_file.read().split()
    index = 0
    for t in range(0, n_tests):
        n_states = int(tr_list[index])
        index = index + 1
        for i in range(0, n_states):
            kl_dev = sqrt((kl_avg - float(tr_list[index]))**2)
            index = index + 1
        for i in range(0, n_states):
            for j in range(0, n_states):
                tran_dev = sqrt((tran_avg - float(tr_list[index]))**2)
                index = index + 1
        init_dev = sqrt((init_avg - float(tr_list[index]))**2)
        index = index + 1
    kl_dev = kl_dev / n_tests
    tran_dev = tran_dev / n_tests
    init_dev = init_dev / n_tests
    in_file.close()
    return kl_avg, kl_dev, tran_avg, tran_dev, init_avg, init_dev



prefix = sys.argv[1]
# prefix = "FC_2_0.1_"

# to compute
eval_err_avg = 0.0
eval_err_dev = 0.0
decod_err_avg = 0.0
decod_err_dev = 0.0
decod_err_path_std_avg = 0.0
decod_err_path_std_dev = 0.0
decod_err_path_compr_avg = 0.0
decod_err_path_compr_dev = 0.0
train_err_statekl_std_avg = 0.0
train_err_statekl_std_dev = 0.0
train_err_transitions_std_avg = 0.0
train_err_transitions_std_dev = 0.0
train_err_initial_std_avg = 0.0
train_err_initial_std_dev = 0.0
train_err_statekl_compr_avg = 0.0
train_err_statekl_compr_dev = 0.0
train_err_transitions_compr_avg = 0.0
train_err_transitions_compr_dev = 0.0
train_err_initial_compr_avg = 0.0
train_err_initial_compr_dev = 0.0
# -- times
eval_time_std_avg = 0.0
eval_time_std_dev = 0.0
eval_time_compr_avg = 0.0
eval_time_compr_dev = 0.0
decod_time_std_avg = 0.0
decod_time_std_dev = 0.0
decod_time_compr_avg = 0.0
decod_time_compr_dev = 0.0
train_time_std_avg = 0.0
train_time_std_dev = 0.0
train_time_compr_avg = 0.0
train_time_compr_dev = 0.0

# Evaluation
eval_err_avg, eval_err_dev = aggregate(f_ev)
eval_time_std_avg, eval_time_std_dev = aggregate(f_ev_std_t)
eval_time_compr_avg, eval_time_compr_dev = aggregate(f_ev_compr_t)

# Decoding
decod_err_avg, decod_err_dev = aggregate(f_de_pr)
decod_err_path_std_avg, decod_err_path_std_dev = aggregate(f_de_std_pa)
decod_err_path_compr_avg, decod_err_path_compr_dev = aggregate(f_de_compr_pa)
decod_time_std_avg, decod_time_std_dev = aggregate(f_de_std_t)
decod_time_compr_avg, decod_time_compr_dev = aggregate(f_de_compr_t)

# Training
train_err_statekl_std_avg, train_err_statekl_std_dev, train_err_transitions_std_avg, train_err_transitions_std_dev, train_err_initial_std_avg, train_err_initial_std_dev = aggregate_measures_training(prefix+f_tr_std)
train_err_statekl_compr_avg, train_err_statekl_compr_dev, train_err_transitions_compr_avg, train_err_transitions_compr_dev, train_err_initial_compr_avg, train_err_initial_compr_dev = aggregate_measures_training(prefix+f_tr_compr)
train_time_std_avg, train_time_std_dev = aggregate(f_tr_std_t)
train_time_compr_avg, train_time_compr_dev = aggregate(f_tr_compr_t)

# print informations
print("=== Errors ===")
print("Evaluation error:",eval_err_avg," (",eval_err_dev,")")
print("Decoding error:",decod_err_avg," (",decod_err_dev,")")
print("Decoding path error STD:",decod_err_path_std_avg," (",decod_err_path_std_dev,")")
print("Decoding path error COMPR:",decod_err_path_compr_avg," (",decod_err_path_compr_dev,")")
print("Training errors STD: StatesKL:",train_err_statekl_std_avg," (",train_err_statekl_std_dev,") Transitions:", train_err_transitions_std_avg," (",train_err_transitions_std_dev,")  Initial:",train_err_initial_std_avg," (",train_err_initial_std_dev,")")
print("Training errors COMPR: StatesKL:",train_err_statekl_compr_avg," (",train_err_statekl_compr_dev,") Transitions:", train_err_transitions_compr_avg," (",train_err_transitions_compr_dev,")  Initial:",train_err_initial_compr_avg," (",train_err_initial_compr_dev,")")
print("=== Execution times (in seconds) ===")
print("Evaluation: STD:",eval_time_std_avg," (",eval_time_std_dev,") COMPR:",eval_time_compr_avg," (",eval_time_compr_dev,")")
print("Decoding: STD:",decod_time_std_avg," (",decod_time_std_dev,") COMPR:",decod_time_compr_avg," (",decod_time_compr_dev,")")
print("Training: STD:",train_time_std_avg," (",train_time_std_dev,") COMPR:",train_time_compr_avg," (",train_time_compr_dev,")")
