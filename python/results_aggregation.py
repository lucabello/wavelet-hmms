#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import utilities_io as uio
from math import sqrt
import sys
from os import listdir
import os
import subprocess
import matplotlib.pyplot as plt
from statistics import median, pstdev

# OPTIONS
n_tests = 100
save_boxplots = True
topologies = ["FC"]
n_states = ["2", "3", "5"]
boxplot_states = ["2", "3", "5"]
etas = ["0.1","0.2","0.3","0.4","0.5","0.6", "0.7", "0.8", "0.9", "1.0"]
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
folder = "plots/"
base_savename = folder+"PLOT_"

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


def save_boxplot(data, title, topology, suffix, ylabel, xlabel):
    f = base_savename + topology + "_" + suffix
    plt.title(title)
    # fig, ax = plt.subplots(len(n_states))
    fig, ax = plt.subplots(len(boxplot_states))
    for i in range(0, len(boxplot_states)):
        ax[i].set(title=boxplot_states[i]+" states")
        ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
            alpha=0.5)
        ax[i].set_xticklabels(etas)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-3,10))
        # remove 3 outliers so boxplots look better
        for count in [0, 1, 2]:
            for eta in range(0, len(data[i])):
                data[i][eta].remove(max(data[i][eta]))
                data[i][eta].remove(min(data[i][eta]))
        ax[i].boxplot(data[i])
        # plt.figure(figsize=(2,1))
        # plt.tight_layout()
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f,format='png')
    print(f,"saved")
    plt.close()
    plt.clf()


def save_plot(x, y_allstates, title, topology, suffix, ylabel, xlabel):
    f = base_savename + topology + "_" + suffix
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    for i in range(0, len(n_states)):
        plt.title(title)
        plt.plot(x, y_allstates[i], label=n_states[i]+" states", \
            linestyle=line_styles[i], marker="o")
        plt.legend(loc='upper right', frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,10))
    plt.savefig(f,format='png')
    print(f,"saved")
    plt.close()
    plt.clf()


def plot_evaluation(topology):
    ev_list = []
    y_all = []
    for n in n_states:
        ev = []
        y = []
        for eta in etas:
            f = "tests/" + topology + "_" + n + "_" + eta + "_"
            list = uio.read_file_to_list(f+f_ev)
            floatlist = [float(e) for e in list]
            y.append(median(floatlist))
            ev.append(floatlist)
        ev_list.append(ev)
        y_all.append(y)
    save_plot(x, y_all, "Evaluation: probability relative error", topology, \
        "evaluation", "Error (relative)", "State separation")
    boxplot_list = []
    boxplot_list.append(ev_list[n_states.index(boxplot_states[0])])
    boxplot_list.append(ev_list[n_states.index(boxplot_states[1])])
    boxplot_list.append(ev_list[n_states.index(boxplot_states[2])])
    save_boxplot(data=boxplot_list, title="Evaluation: probability error", \
        topology=topology, suffix="evaluation_boxplot", \
        ylabel="Error (relative)", xlabel="State separation")


def plot_decoding(topology):
    de_list = []
    y_all = []
    for n in n_states:
        de = []
        y = []
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
            y.append(median(floatlist_diff))
        de_list.append(de)
        y_all.append(y)
    save_plot(x, y_all, "Decoding: relative path error", topology, \
        "decoding", "Error (relative)", "State separation")
    boxplot_list = []
    boxplot_list.append(de_list[n_states.index(boxplot_states[0])])
    boxplot_list.append(de_list[n_states.index(boxplot_states[1])])
    boxplot_list.append(de_list[n_states.index(boxplot_states[2])])
    save_boxplot(data=boxplot_list, title="Decoding: path error increment", \
        topology=topology, suffix="decoding_boxplot", \
        ylabel="Error (relative)", xlabel="State separation")


def summarize_training(filename):
    kl = []
    tr = []
    pi = []
    iters = []
    list = uio.read_file_to_list(filename)
    index = 0
    for t in range(0, n_tests):
        statesnum = int(list[index])
        index = index + 1
        # average state kl
        current_kl = 0
        for i in range(0, statesnum):
            current_kl = current_kl + float(list[index])
            index = index + 1
        current_kl = current_kl/statesnum
        kl.append(current_kl)
        # average transition relative error
        current_tr = 0
        for i in range(0, statesnum):
            for j in range(0, statesnum):
                current_tr = current_tr + float(list[index])
                index = index + 1
        current_tr = current_tr / (statesnum**2)
        tr.append(current_tr)
        # average initial distribution relative error
        pi.append(float(list[index]))
        index = index + 1
        iters.append(int(list[index]))
        index = index + 1
    return kl, tr, pi, iters


def plot_training(topology):
    tr_kl_list = []
    tr_tr_list = []
    tr_in_list = []
    y_kl_all = []
    y_tr_all = []
    y_in_all = []
    y_iters_std_all = []
    y_iters_compr_all = []
    y_iters_all = []
    for n in n_states:
        tr_kl = []
        tr_tr = []
        tr_in = []
        y_kl = []
        y_tr = []
        y_in = []
        y_iters_std = []
        y_iters_compr = []
        y_iters = []
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
            tr_iters_std = []
            tr_iters_compr = []
            tr_iters_diff = []
            f = "tests/" + topology + "_" + n + "_" + eta + "_"
            # read standard training results
            tr_kl_std, tr_tr_std, tr_in_std, tr_iters_std = summarize_training(f +
                f_tr_std)
            # read compressed training results
            tr_kl_compr, tr_tr_compr, tr_in_compr, tr_iters_compr = summarize_training(f +
                f_tr_compr)
            for i in range(0, len(tr_kl_std)):
                tr_kl_diff.append(tr_kl_compr[i] - tr_kl_std[i])
            for i in range(0, len(tr_tr_std)):
                tr_tr_diff.append(tr_tr_compr[i] - tr_tr_std[i])
            for i in range(0, len(tr_in_std)):
                tr_in_diff.append(tr_in_compr[i] - tr_in_std[i])
            for i in range(0, len(tr_iters_std)):
                tr_iters_std.append(tr_iters_std[i])
                tr_iters_compr.append(tr_iters_compr[i])
                tr_iters_diff.append(tr_iters_compr[i] - tr_iters_std[i])
            tr_kl.append(tr_kl_diff)
            tr_tr.append(tr_tr_diff)
            tr_in.append(tr_in_diff)
            y_kl.append(median(tr_kl_diff))
            y_tr.append(median(tr_tr_diff))
            y_in.append(median(tr_in_diff))
            y_iters_std.append(median(tr_iters_std))
            y_iters_compr.append(median(tr_iters_compr))
            y_iters.append(median(tr_iters_diff))

        tr_kl_list.append(tr_kl)
        tr_tr_list.append(tr_tr)
        tr_in_list.append(tr_in)
        y_kl_all.append(y_kl)
        y_tr_all.append(y_tr)
        y_in_all.append(y_in)
        y_iters_std_all.append(y_iters_std)
        y_iters_compr_all.append(y_iters_compr)
        y_iters_all.append(y_iters)

    save_plot(x, y_kl_all, "Training: KL-divergence difference", topology,
        "training_kl", "Difference (abs)", "State separation")
    save_plot(x, y_tr_all, "Training: Transitions relative error difference",
        topology, "training_tr", "Difference (abs)", "State separation")
    save_plot(x, y_in_all, "Training: Initial distribution error difference",
        topology, "training_in", "Difference (abs)", "State separation")

    save_plot(x, y_iters_std_all, "Training: Iterations for standard algorithm",
        topology, "training_iters_std", "Iterations", "State separation")
    save_plot(x, y_iters_compr_all, "Training: Iterations for compr. algorithm",
        topology, "training_iters_compr", "Iterations", "State separation")
    save_plot(x, y_iters_all, "Training: Iterations difference",
        topology, "training_iters_diff", "Iteration Difference",
        "State separation")

    # Training KL
    boxplot_kl_list = []
    boxplot_kl_list.append(tr_kl_list[n_states.index(boxplot_states[0])])
    boxplot_kl_list.append(tr_kl_list[n_states.index(boxplot_states[1])])
    boxplot_kl_list.append(tr_kl_list[n_states.index(boxplot_states[2])])
    save_boxplot(data=boxplot_kl_list, title="Training: states error", \
        topology=topology, suffix="training_kl_boxplot", \
        ylabel="Difference (abs)", xlabel="State separation")
    # Training TR
    boxplot_tr_list = []
    boxplot_tr_list.append(tr_tr_list[n_states.index(boxplot_states[0])])
    boxplot_tr_list.append(tr_tr_list[n_states.index(boxplot_states[1])])
    boxplot_tr_list.append(tr_tr_list[n_states.index(boxplot_states[2])])
    save_boxplot(data=boxplot_tr_list, title="Training: transitions error", \
        topology=topology, suffix="training_tr_boxplot", \
        ylabel="Difference (abs)", xlabel="State separation")
    # Training IN
    boxplot_in_list = []
    boxplot_in_list.append(tr_in_list[n_states.index(boxplot_states[0])])
    boxplot_in_list.append(tr_in_list[n_states.index(boxplot_states[1])])
    boxplot_in_list.append(tr_in_list[n_states.index(boxplot_states[2])])
    save_boxplot(data=boxplot_in_list, \
        title="Training: initial distribution error", topology=topology, \
        suffix="training_in_boxplot", ylabel="Difference (abs)", \
        xlabel="State separation")



def plot_speedup(topology):
    titles = ["Speedup: evaluation", "Speedup: decoding", "Speedup: training"]
    problems = ["evaluation", "decoding", "training"]
    files = dict()
    files["evaluation"] = [f_ev_std_t, f_ev_compr_t]
    files["decoding"] =  [f_de_std_t, f_de_compr_t]
    files["training"] = [f_tr_std_t, f_tr_compr_t]
    for p in problems:
        speedup_list = []
        y_all = []
        for n in n_states:
            speedup = []
            y = []
            for eta in etas:
                f = "tests/" + topology + "_" + n + "_" + eta + "_"
                list = uio.read_file_to_list(f+files[p][0])
                floatlist_std = [float(e) for e in list]
                list = uio.read_file_to_list(f+files[p][1])
                floatlist_compr = [float(e) for e in list]
                floatlist_ratio = []
                for i in range(0, len(floatlist_std)):
                    floatlist_ratio.append(floatlist_std[i]/floatlist_compr[i])
                speedup.append(floatlist_ratio)
                y.append(median(floatlist_ratio))
            speedup_list.append(speedup)
            y_all.append(y)
        save_plot(x, y_all, "Speedup: "+p, topology, "speedup_"+p, \
            "Speedup", "State separation")
        boxplot_speedup_list = []
        boxplot_speedup_list.append( \
            speedup_list[n_states.index(boxplot_states[0])])
        boxplot_speedup_list.append( \
            speedup_list[n_states.index(boxplot_states[1])])
        boxplot_speedup_list.append( \
            speedup_list[n_states.index(boxplot_states[2])])
        save_boxplot(data=boxplot_speedup_list, title="Speedup: "+p, \
            topology=topology, suffix="speedup_"+p+"_boxplot", \
            ylabel="Speedup", xlabel="State separation")

if __name__ == "__main__":
    # create folder if it doesnt exist
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    # produce plots
    for topology in topologies:
        f = base_savename + topology + "_"
        plot_evaluation(topology)
        plot_decoding(topology)
        plot_training(topology)
        plot_speedup(topology)
    # join plots with corresponding boxplot
    files = listdir(folder)
    for f in files:
        if "boxplot" not in f and "MERGE" not in f:
            subprocess.call("convert "+folder+f+" "+folder+f+ \
                "_boxplot +append " + \
                folder+"MERGE_"+f[5:], shell=True)
            print(f," merged")
