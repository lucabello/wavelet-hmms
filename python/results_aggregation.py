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
n_tests = 100 # to know # of lines in training reuslts
f_ev = "evaluation"
f_de_pr = "decoding_prob"
f_de_std_pa = "decoding_std_path"
f_de_compr_pa = "decoding_compr_path"
f_tr_std = "training_std"
f_tr_compr = "training_compr"

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

def aggregate_measures_training(filename):
    # calculate std avg
    mean_avg = 0.0
    mean_dev = 0.0
    dev_avg = 0.0
    dev_dev = 0.0
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
            mean_avg = mean_avg + float(tr_list[index])
            index = index + 1
            dev_avg = dev_avg + float(tr_list[index])
            index = index + 1
        for i in range(0, n_states):
            for j in range(0, n_states):
                tran_avg = tran_avg + float(tr_list[index])
                index = index + 1
        for i in range(0, n_states):
            init_avg = init_avg + float(tr_list[index])
            index = index + 1
    mean_avg = mean_avg / n_tests
    dev_avg = dev_avg / n_tests
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
            mean_dev = sqrt((mean_avg - float(tr_list[index]))**2)
            index = index + 1
            dev_dev = sqrt((dev_avg - float(tr_list[index]))**2)
            index = index + 1
        for i in range(0, n_states):
            for j in range(0, n_states):
                tran_dev = sqrt((tran_avg - float(tr_list[index]))**2)
                index = index + 1
        for i in range(0, n_states):
            init_dev = sqrt((init_avg - float(tr_list[index]))**2)
            index = index + 1
    mean_dev = mean_dev / n_tests
    dev_dev = dev_dev / n_tests
    tran_dev = tran_dev / n_tests
    init_dev = init_dev / n_tests
    in_file.close()
    return mean_avg, mean_dev, dev_avg, dev_dev, tran_avg, tran_dev, init_avg, init_dev



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
train_err_statemean_std_avg = 0.0
train_err_statemean_std_dev = 0.0
train_err_statedev_std_avg = 0.0
train_err_statedev_std_dev = 0.0
train_err_transitions_std_avg = 0.0
train_err_transitions_std_dev = 0.0
train_err_initial_std_avg = 0.0
train_err_initial_std_dev = 0.0
train_err_statemean_compr_avg = 0.0
train_err_statemean_compr_dev = 0.0
train_err_statedev_compr_avg = 0.0
train_err_statedev_compr_dev = 0.0
train_err_transitions_compr_avg = 0.0
train_err_transitions_compr_dev = 0.0
train_err_initial_compr_avg = 0.0
train_err_initial_compr_dev = 0.0

# Evaluation
in_file = open(prefix+f_ev, "r")
eval_list = in_file.read().split()
in_file.close()
eval_err_avg, eval_err_dev = aggregate_measures(eval_list)

# Decoding
in_file = open(prefix+f_de_pr, "r")
decod_list = in_file.read().split()
in_file.close()
decod_err_avg, decod_err_dev = aggregate_measures(decod_list)
in_file = open(prefix+f_de_std_pa, "r")
path_std_list = in_file.read().split()
in_file.close()
decod_err_path_std_avg, decod_err_path_std_dev = aggregate_measures(path_std_list)
in_file = open(prefix+f_de_compr_pa, "r")
path_compr_list = in_file.read().split()
in_file.close()
decod_err_path_compr_avg, decod_err_path_compr_dev = aggregate_measures(path_compr_list)

# Training
train_err_statemean_std_avg, train_err_statemean_std_dev, train_err_statedev_std_avg, train_err_statedev_std_dev, train_err_transitions_std_avg, train_err_transitions_std_dev, train_err_initial_std_avg, train_err_initial_std_dev = aggregate_measures_training(prefix+f_tr_std)
train_err_statemean_compr_avg, train_err_statemean_compr_dev, train_err_statedev_compr_avg, train_err_statedev_compr_dev, train_err_transitions_compr_avg, train_err_transitions_compr_dev, train_err_initial_compr_avg, train_err_initial_compr_dev = aggregate_measures_training(prefix+f_tr_compr)

# print informations
print("Evaluation error:",eval_err_avg," (",eval_err_dev,")")
print("Decoding error:",decod_err_avg," (",decod_err_dev,")")
print("Decoding path error STD:",decod_err_path_std_avg," (",decod_err_path_std_dev,")")
print("Decoding path error COMPR:",decod_err_path_compr_avg," (",decod_err_path_compr_dev,")")
print("Training errors STD: States: [",train_err_statemean_std_avg," (",train_err_statemean_std_dev,")  ",train_err_statedev_std_avg," (",train_err_statedev_std_dev,") ]  Transitions:", train_err_transitions_std_avg," (",train_err_transitions_std_dev,")  Initial:",train_err_initial_std_avg," (",train_err_initial_std_dev,")")
print("Training errors COMPR: States: [",train_err_statemean_compr_avg," (",train_err_statemean_compr_dev,")  ",train_err_statedev_compr_avg," (",train_err_statedev_compr_dev,") ]  Transitions:", train_err_transitions_compr_avg," (",train_err_transitions_compr_dev,")  Initial:",train_err_initial_compr_avg," (",train_err_initial_compr_dev,")")