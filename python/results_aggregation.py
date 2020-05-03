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
save_boxplots = True
n_states = ["2", "3", "5"]
etas = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

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
    element_list = [float(e) for e in element_list]
    result_avg = median(element_list)
    result_dev = pstdev(element_list)
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
        statesnum = int(tr_list[index])
        index = index + 1
        current_kl = 0
        for i in range(0, statesnum):
            current_kl = current_kl + float(tr_list[index])
            index = index + 1
        current_kl = current_kl/statesnum
        kl_list.append(current_kl)
        current_tr = 0
        for i in range(0, statesnum):
            for j in range(0, statesnum):
                current_tr = current_tr + float(tr_list[index])
                index = index + 1
        current_tr = current_tr / (statesnum**2)
        tran_list.append(current_tr)
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
    # print("=== Errors for",prefix,"===")
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


def single_boxplot(data, title, topology, suffix, ybot, ytop):
    base_savename = "graphs/GRAPH_" + topology + "_"
    plt.title(title)
    fig, ax = plt.subplots(len(n_states), constrained_layout=True)
    for i in range(0, len(n_states)):
        ax[i].set(title=n_states[i]+" states")
        ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax[i].set_ylim(ybot[i], ytop[i])
        ax[i].set_xticklabels(etas)
        ax[i].boxplot(data[i])
        # plt.figure(figsize=(2,1))
        # plt.tight_layout()
    plt.savefig(base_savename+suffix)
    plt.clf()


def produce_boxplots(topology):
    ev_list = []
    # Evaluation
    for n in n_states:
        ev = []
        for eta in etas:
            f = "tests/" + topology + "_" + n + "_" + eta + "_"
            list = uio.read_file_to_list(f+f_ev)
            floatlist = [float(e) for e in list]
            ev.append(floatlist)
        ev_list.append(ev)
    if topology == "LR":
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.001, 0.001, 0.001]
    else:
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.003, 0.003, 0.003]
    single_boxplot(data=ev_list, title="Evaluation: probability error", \
        topology=topology, suffix="evaluation_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)
    # Decoding
    de_list = []
    for n in n_states:
        de = []
        for eta in etas:
            f = "tests/" + topology + "_" + n + "_" + eta + "_"
            list = uio.read_file_to_list(f+f_de_std_pa)
            floatlist_std = [float(e) for e in list]
            list = uio.read_file_to_list(f+f_de_compr_pa)
            floatlist_compr = [float(e) for e in list]
            floatlist_diff = []
            for i in range(0, len(floatlist_std)):
                floatlist_diff.append(floatlist_compr[i] - floatlist_std[i])
            de.append(floatlist_diff)
        de_list.append(de)
    if topology == "LR":
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.001, 0.001, 0.001]
    else:
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.001, 0.001, 0.001]
    single_boxplot(data=de_list, title="Decoding: path error increment", \
        topology=topology, suffix="decoding_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)
    # Training
    tr_kl_list = []
    tr_tr_list = []
    tr_in_list = []
    for n in n_states:
        tr_kl = []
        tr_tr = []
        tr_in = []
        for eta in etas:
            tr_kl_std = []
            tr_tr_std = []
            tr_in_std = []
            tr_kl_compr = []
            tr_tr_compr = []
            tr_in_compr = []
            tr_kl_diff = []
            tr_tr_diff = []
            tr_in_diff = []
            f = "tests/" + topology + "_" + n + "_" + eta + "_"
            in_file = open(f+f_tr_std, "r")
            tr_list = in_file.read().split()
            index = 0
            for t in range(0, n_tests):
                statesnum = int(tr_list[index])
                index = index + 1
                current_kl = 0
                for i in range(0, statesnum):
                    current_kl = current_kl + float(tr_list[index])
                    index = index + 1
                current_kl = current_kl/statesnum
                tr_kl_std.append(current_kl)
                current_tr = 0
                for i in range(0, statesnum):
                    for j in range(0, statesnum):
                        current_tr = current_tr + float(tr_list[index])
                        index = index + 1
                current_tr = current_tr / (statesnum**2)
                tr_tr_std.append(current_tr)
                tr_in_std.append(float(tr_list[index]))
                index = index + 1
            in_file.close()
            in_file = open(f+f_tr_compr, "r")
            tr_list = in_file.read().split()
            index = 0
            for t in range(0, n_tests):
                statesnum = int(tr_list[index])
                index = index + 1
                for i in range(0, statesnum):
                    tr_kl_compr.append(float(tr_list[index]))
                    index = index + 1
                for i in range(0, statesnum):
                    for j in range(0, statesnum):
                        tr_tr_compr.append(float(tr_list[index]))
                        index = index + 1
                tr_in_compr.append(float(tr_list[index]))
                index = index + 1
            in_file.close()
            for i in range(0, len(tr_kl_std)):
                tr_kl_diff.append(tr_kl_compr[i] - tr_kl_std[i])
            for i in range(0, len(tr_tr_std)):
                tr_tr_diff.append(tr_tr_compr[i] - tr_tr_std[i])
            for i in range(0, len(tr_in_std)):
                tr_in_diff.append(tr_in_compr[i] - tr_in_std[i])
            tr_kl.append(tr_kl_diff)
            tr_tr.append(tr_tr_diff)
            tr_in.append(tr_in_diff)
        tr_kl_list.append(tr_kl)
        tr_tr_list.append(tr_tr)
        tr_in_list.append(tr_in)

    # Training KL
    if topology == "CI":
        tmp_ybot = [-0.05, -1, -2]
        tmp_ytop = [0.01, 0.3, 1.5]
    elif topology == "FC":
        tmp_ybot = [-0.02, -1, -5]
        tmp_ytop = [0.01, 0.3, 3]
    else:
        tmp_ybot = [-0.05, -8, -14]
        tmp_ytop = [0.01, 5, 10]
    single_boxplot(data=tr_kl_list, title="Training: states error", \
        topology=topology, suffix="training_kl_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)
    # Training TR
    single_boxplot(data=tr_tr_list, title="Training: transitions error", \
        topology=topology, suffix="training_tr_boxplot", ybot=[-1.5, -1.5, -1.5], \
        ytop=[1.5, 1.5, 1.5])
    # Training IN
    if topology == "LR":
        tmp_ybot = [-0.000011, -0.000011, -0.000011]
        tmp_ytop = [0.0000005, 0.0000005, 0.0000005]
    else:
        tmp_ybot = [-0.00000025, -0.000005, -0.00001]
        tmp_ytop = [0.00000005, 0.0000005, 0.0000005]
    single_boxplot(data=tr_in_list, title="Training: initial distribution error", \
        topology=topology, suffix="training_in_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)



if __name__ == "__main__":
    # prefix = sys.argv[1]
    # prefix = "FC_2_0.1_"
    folder = "graphs/"
    topologies = ["FC", "CI", "LR"]
    n_states = ["2", "3", "5"]
    etas = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    for topology in topologies:
        produce_boxplots(topology)
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        y5 = []
        y6e = []
        y6d = []
        y6t = []
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
            speedup_ev_list = []
            speedup_de_list = []
            speedup_tr_list = []
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
                de_path_compr_list.append(de_path_compr)
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
                speedup_ev_list.append(ev_t_std/ev_t_compr)
                speedup_de_list.append(de_t_std/de_t_compr)
                speedup_tr_list.append(tr_t_std/tr_t_compr)
                speedup_list.append( (ev_t_std+de_t_std+tr_t_std) / (ev_t_compr+de_t_compr+tr_t_compr) )
                tr_kl_list.append(tr_kl_compr - tr_kl_std)
                tr_tr_list.append(tr_tr_compr - tr_tr_std)
                tr_in_list.append(tr_in_compr - tr_in_std)

            f = folder + topology + "_" + n_states[i]+"_"
            # uio.write_list(f+f_ev, ev_list) # used
            # uio.write_list(f+f_de_pr, de_list)
            # uio.write_list(f+f_de_std_pa, de_path_std_list)
            # uio.write_list(f+f_de_compr_pa, de_path_compr_list)
            # uio.write_list(f+f_tr_std+"_kl", tr_kl_std_list)
            # uio.write_list(f+f_tr_std+"_tr", tr_tr_std_list)
            # uio.write_list(f+f_tr_std+"_in", tr_in_std_list)
            # uio.write_list(f+f_tr_compr+"_kl", tr_kl_compr_list)
            # uio.write_list(f+f_tr_compr+"_tr", tr_tr_compr_list)
            # uio.write_list(f+f_tr_compr+"_in", tr_in_compr_list)
            # uio.write_list(f+f_ev_std_t, ev_t_std_list)
            # uio.write_list(f+f_ev_compr_t, ev_t_compr_list)
            # uio.write_list(f+f_de_std_t, de_t_std_list)
            # uio.write_list(f+f_de_compr_t, de_t_compr_list)
            # uio.write_list(f+f_tr_std_t, tr_t_std_list)
            # uio.write_list(f+f_tr_compr_t, tr_t_compr_list)
            # uio.write_list(f+"decoding_path", de_path_diff_list) # used
            # uio.write_list(f+"speedup", speedup_list)
            # uio.write_list(f+"training_kl", tr_kl_list) # used
            # uio.write_list(f+"training_tr", tr_tr_list) # used
            # uio.write_list(f+"training_in", tr_in_list) # used
            # uio.write_list(f+"speedup_evaluation", speedup_ev_list) # used
            # uio.write_list(f+"speedup_decoding", speedup_de_list) # used
            # uio.write_list(f+"speedup_training", speedup_tr_list) # used


            y1.append(ev_list)
            y2.append(de_path_diff_list)
            y3.append(tr_kl_list)
            y4.append(tr_tr_list)
            y5.append(tr_in_list)
            y6e.append(speedup_ev_list)
            y6d.append(speedup_de_list)
            y6t.append(speedup_tr_list)



        # print data to plot as debug output
        # print("== Graph Data ==")
        # print("Evaluation:", y1)
        # print("Decoding:", y2)
        # print("Training KL:", y3)
        # print("Training transitions:", y4)
        # print("Training initial:", y5)
        # print("Speedup:", y6)

        f = folder + "GRAPH_" + topology + "_"
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        line_styles = ['-', '--', ':']

        # Evaluation
        for i in range(0, len(n_states)):
            plt.title("Evaluation: probability error")
            plt.plot(x, y1[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y1)[0]-abs(min(y1)[0]/10), max(y1)[0]+abs(max(y1)[0]/10))
        plt.savefig(f+"evaluation")
        plt.clf()
        # Decoding
        for i in range(0, len(n_states)):
            plt.title("Decoding: path error increment")
            plt.plot(x, y2[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y2)[0]-abs(min(y2)[0]/10), max(y2)[0]+abs(max(y2)[0]/10))
        plt.savefig(f+"decoding")
        plt.clf()
        # Training
        for i in range(0, len(n_states)):
            plt.title("Training: states error")
            plt.plot(x, y3[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y3)[0]-abs(min(y3)[0]/10), max(y3)[0]+abs(max(y3)[0]/10))
        plt.savefig(f+"training_kl")
        plt.clf()
        for i in range(0, len(n_states)):
            plt.title("Training: transitions error")
            plt.plot(x, y4[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(-0.002, 0.0005)
        plt.savefig(f+"training_tr")
        plt.clf()
        for i in range(0, len(n_states)):
            plt.title("Training: initial distribution error")
            plt.plot(x, y5[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y5)[0]-abs(min(y5)[0]/10), max(y5)[0]+abs(max(y5)[0]/10))
        plt.savefig(f+"training_in")
        plt.clf()
        # Speedup
        for i in range(0, len(n_states)):
            plt.title("Evaluation: Speedup")
            plt.plot(x, y6e[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y6)[0]-10, max(y6)[0]+abs(max(y6)[0]/10))
        plt.savefig(f+"speedup_evaluation")
        plt.clf()
        for i in range(0, len(n_states)):
            plt.title("Decoding: Speedup")
            plt.plot(x, y6d[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y6)[0]-10, max(y6)[0]+abs(max(y6)[0]/10))
        plt.savefig(f+"speedup_decoding")
        plt.clf()
        for i in range(0, len(n_states)):
            plt.title("Training: Speedup")
            plt.plot(x, y6t[i], label=n_states[i]+" states", linestyle=line_styles[i], marker="o")
            plt.legend(loc='upper right', frameon=False)
            #plt.ylim(min(y6)[0]-10, max(y6)[0]+abs(max(y6)[0]/10))
        plt.savefig(f+"speedup_training")
        plt.clf()
