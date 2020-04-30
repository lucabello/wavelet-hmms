#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import utilities_io as uio
from math import sqrt
import sys
import matplotlib.pyplot as plt
from statistics import median, pstdev

def print(*args):
    __builtins__.print(*("%.2e" % a if isinstance(a, float) else a
                         for a in args))


# OPTIONS
n_tests = 100

# file names
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
        result_dev = result_dev + ((ef - result_avg)**2)
    result_dev = sqrt(result_dev) / len(element_list)
    return result_avg, result_dev

def aggregate(prefix, suffix):
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
    kl_list = []
    tran_list = []
    init_list = []
    in_file = open(filename, "r")
    tr_list = in_file.read().split()
    index = 0
    for t in range(0, n_tests):
        n_states = int(tr_list[index])
        index = index + 1
        for i in range(0, n_states):
            kl_list.append(float(tr_list[index]))
            index = index + 1
        for i in range(0, n_states):
            for j in range(0, n_states):
                tran_list.append(float(tr_list[index]))
                index = index + 1
        init_list.append(float(tr_list[index]))
        index = index + 1
    kl_avg = median(kl_list)
    tran_avg = median(tran_list)
    init_avg = median(init_list)
    in_file.close()
    # calculate std dev
    kl_dev = pstdev(kl_list)
    tran_dev = pstdev(tran_list)
    init_dev = pstdev(init_list)
    return kl_avg, kl_dev, tran_avg, tran_dev, init_avg, init_dev


def compute_statistics(prefix):
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
    eval_err_avg, eval_err_dev = aggregate(prefix, f_ev)
    eval_time_std_avg, eval_time_std_dev = aggregate(prefix, f_ev_std_t)
    eval_time_compr_avg, eval_time_compr_dev = aggregate(prefix, f_ev_compr_t)

    # Decoding
    decod_err_avg, decod_err_dev = aggregate(prefix, f_de_pr)
    decod_err_path_std_avg, decod_err_path_std_dev = aggregate(prefix, f_de_std_pa)
    decod_err_path_compr_avg, decod_err_path_compr_dev = aggregate(prefix, f_de_compr_pa)
    decod_time_std_avg, decod_time_std_dev = aggregate(prefix, f_de_std_t)
    decod_time_compr_avg, decod_time_compr_dev = aggregate(prefix, f_de_compr_t)

    # Training
    train_err_statekl_std_avg, train_err_statekl_std_dev, train_err_transitions_std_avg, train_err_transitions_std_dev, train_err_initial_std_avg, train_err_initial_std_dev = aggregate_measures_training(prefix+f_tr_std)
    train_err_statekl_compr_avg, train_err_statekl_compr_dev, train_err_transitions_compr_avg, train_err_transitions_compr_dev, train_err_initial_compr_avg, train_err_initial_compr_dev = aggregate_measures_training(prefix+f_tr_compr)
    train_time_std_avg, train_time_std_dev = aggregate(prefix, f_tr_std_t)
    train_time_compr_avg, train_time_compr_dev = aggregate(prefix, f_tr_compr_t)

    # print informations
    # print("=== Errors ===")
    # print("Evaluation error:",eval_err_avg," (",eval_err_dev,")")
    # print("Decoding error:",decod_err_avg," (",decod_err_dev,")")
    # print("Decoding path error STD:",decod_err_path_std_avg," (",decod_err_path_std_dev,")")
    # print("Decoding path error COMPR:",decod_err_path_compr_avg," (",decod_err_path_compr_dev,")")
    # print("Training errors STD: StatesKL:",train_err_statekl_std_avg," (",train_err_statekl_std_dev,") Transitions:", train_err_transitions_std_avg," (",train_err_transitions_std_dev,")  Initial:",train_err_initial_std_avg," (",train_err_initial_std_dev,")")
    # print("Training errors COMPR: StatesKL:",train_err_statekl_compr_avg," (",train_err_statekl_compr_dev,") Transitions:", train_err_transitions_compr_avg," (",train_err_transitions_compr_dev,")  Initial:",train_err_initial_compr_avg," (",train_err_initial_compr_dev,")")
    # print("=== Execution times (in seconds) ===")
    # print("Evaluation: STD:",eval_time_std_avg," (",eval_time_std_dev,") COMPR:",eval_time_compr_avg," (",eval_time_compr_dev,")")
    # print("Decoding: STD:",decod_time_std_avg," (",decod_time_std_dev,") COMPR:",decod_time_compr_avg," (",decod_time_compr_dev,")")
    # print("Training: STD:",train_time_std_avg," (",train_time_std_dev,") COMPR:",train_time_compr_avg," (",train_time_compr_dev,")")

    return eval_err_avg, decod_err_avg, decod_err_path_std_avg, \
        decod_err_path_compr_avg, train_err_statekl_std_avg, \
         train_err_transitions_std_avg, train_err_initial_std_avg, \
         train_err_statekl_compr_avg, train_err_transitions_compr_avg, \
         train_err_initial_compr_avg, eval_time_std_avg, eval_time_compr_avg, \
         decod_time_std_avg, decod_time_compr_avg, train_time_std_avg, \
         train_time_compr_avg

y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
if __name__ == "__main__":
    # prefix = sys.argv[1]
    # prefix = "FC_2_0.1_"
    folder = "graphs/"
    topology = "FC"
    n_states = ["2", "3", "5"]
    etas = ["1.0", "0.9", "0.8"]#, "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    for i in range(0, len(n_states)):
        ev_list = []
        de_list = []
        de_path_std_list = []
        de_path_compr_list = []
        tr_kl_std_list = []
        tr_tr_std_list = []
        tr_in_std_list = []
        tr_kl_compr_list = []
        tr_tr_compr_list = []
        tr_in_compr_list = []
        ev_t_std_list = []
        ev_t_compr_list = []
        de_t_std_list = []
        de_t_compr_list = []
        tr_t_std_list = []
        tr_t_compr_list = []
        de_path_diff_list = []
        speedup_list = []
        tr_kl_list = []
        tr_tr_list = []
        tr_in_list = []

        for j in range(0, len(etas)):
            prefix = "tests/" + topology + "_" + n_states[i]+"_"+etas[j]+"_"
            ev, de, de_path_std, de_path_compr, tr_kl_std, tr_tr_std, \
                tr_in_std, tr_kl_compr, tr_tr_compr, tr_in_compr, ev_t_std, \
                ev_t_compr, de_t_std, de_t_compr, tr_t_std, \
                tr_t_compr = compute_statistics(prefix)

            ev_list.append(ev)
            de_list.append(de)
            de_path_std_list.append(de_path_std)
            de_path_compr_list.append(de_path_compr_list)
            tr_kl_std_list.append(tr_kl_std)
            tr_tr_std_list.append(tr_tr_std)
            tr_in_std_list.append(tr_in_std)
            tr_kl_compr_list.append(tr_kl_compr)
            tr_tr_compr_list.append(tr_tr_compr)
            tr_in_compr_list.append(tr_in_compr)
            ev_t_std_list.append(ev_t_std)
            ev_t_compr_list.append(ev_t_compr)
            de_t_std_list.append(de_t_std)
            de_t_compr_list.append(de_t_compr)
            tr_t_std_list.append(tr_t_std)
            tr_t_compr_list.append(tr_t_compr)
            de_path_diff_list.append(de_path_compr - de_path_std)
            speedup_list.append( (ev_t_std+de_t_std+tr_t_std) / (ev_t_compr+de_t_compr+tr_t_compr) )
            tr_kl_list.append(tr_kl_compr - tr_kl_std)
            tr_tr_list.append(tr_tr_compr - tr_tr_std)
            tr_in_list.append(tr_in_compr - tr_in_std)

        f = folder + topology + "_" + n_states[i]+"_"
        uio.write_list(f+f_ev, ev_list) # used
        uio.write_list(f+f_de_pr, de_list)
        uio.write_list(f+f_de_std_pa, de_path_std_list)
        uio.write_list(f+f_de_compr_pa, de_path_compr_list)
        uio.write_list(f+f_tr_std+"_kl", tr_kl_std_list)
        uio.write_list(f+f_tr_std+"_tr", tr_tr_std_list)
        uio.write_list(f+f_tr_std+"_in", tr_in_std_list)
        uio.write_list(f+f_tr_compr+"_kl", tr_kl_compr_list)
        uio.write_list(f+f_tr_compr+"_tr", tr_tr_compr_list)
        uio.write_list(f+f_tr_compr+"_in", tr_in_compr_list)
        uio.write_list(f+f_ev_std_t, ev_t_std_list)
        uio.write_list(f+f_ev_compr_t, ev_t_compr_list)
        uio.write_list(f+f_de_std_t, de_t_std_list)
        uio.write_list(f+f_de_compr_t, de_t_compr_list)
        uio.write_list(f+f_tr_std_t, tr_t_std_list)
        uio.write_list(f+f_tr_compr_t, tr_t_compr_list)
        uio.write_list(f+"decoding_path", de_path_diff_list) # used
        uio.write_list(f+"speedup", speedup_list) # used
        uio.write_list(f+"training_kl", tr_kl_list) # used
        uio.write_list(f+"training_tr", tr_tr_list) # used
        uio.write_list(f+"training_in", tr_in_list) # used

        y1.append(ev_list)
        y2.append(de_path_diff_list)
        y3.append(tr_kl_list)
        y4.append(tr_tr_list)
        y5.append(tr_in_list)
        y6.append(speedup_list)


    # print data to plot as debug output
    print("== Graph Data ==")
    print("Evaluation:", y1)
    print("Decoding:", y2)
    print("Training KL:", y3)
    print("Training transitions:", y4)
    print("Training initial:", y5)
    print("Speedup:", y6)

    f = folder + "GRAPH_"
    x = [1.0, 0.9, 0.8]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(0, len(n_states)):
        plt.plot(x, y1[i], label=n_states[i]+" states")
        plt.legend(loc='upper right', frameon=False)
        #plt.ylim(min(y1)[0]-abs(min(y1)[0]/10), max(y1)[0]+abs(max(y1)[0]/10))
    plt.savefig(f+"evaluation")
    plt.clf()
    for i in range(0, len(n_states)):
        plt.plot(x, y2[i], label=n_states[i]+" states")
        plt.legend(loc='upper right', frameon=False)
        #plt.ylim(min(y2)[0]-abs(min(y2)[0]/10), max(y2)[0]+abs(max(y2)[0]/10))
    plt.savefig(f+"decoding")
    plt.clf()
    for i in range(0, len(n_states)):
        plt.plot(x, y3[i], label=n_states[i]+" states")
        plt.legend(loc='upper right', frameon=False)
        #plt.ylim(min(y3)[0]-abs(min(y3)[0]/10), max(y3)[0]+abs(max(y3)[0]/10))
    plt.savefig(f+"training_kl")
    plt.clf()
    for i in range(0, len(n_states)):
        plt.plot(x, y4[i], label=n_states[i]+" states")
        plt.legend(loc='upper right', frameon=False)
        #plt.ylim(-0.002, 0.0005)
    plt.savefig(f+"training_tr")
    plt.clf()
    for i in range(0, len(n_states)):
        plt.plot(x, y5[i], label=n_states[i]+" states")
        plt.legend(loc='upper right', frameon=False)
        #plt.ylim(min(y5)[0]-abs(min(y5)[0]/10), max(y5)[0]+abs(max(y5)[0]/10))
    plt.savefig(f+"training_in")
    plt.clf()
    for i in range(0, len(n_states)):
        plt.plot(x, y6[i], label=n_states[i]+" states")
        plt.legend(loc='upper right', frameon=False)
        #plt.ylim(min(y6)[0]-10, max(y6)[0]+abs(max(y6)[0]/10))
    plt.savefig(f+"speedup")
    plt.clf()
