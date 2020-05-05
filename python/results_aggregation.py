#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import utilities_io as uio
from math import sqrt
import sys
from os import listdir
import subprocess
import matplotlib.pyplot as plt
from statistics import median, pstdev

# OPTIONS
n_tests = 100
save_boxplots = True
topologies = ["FC", "CI", "LR"]
n_states = ["2", "3", "5"]
etas = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
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


def save_boxplot(data, title, topology, suffix, ybot, ytop):
    f = base_savename + topology + "_" + suffix
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
    plt.savefig(f)
    plt.close()
    plt.clf()


def save_plot(x, y_allstates, title, topology, suffix, legendLoc='upper right'):
    f = base_savename + topology + "_" + suffix
    line_styles = ['-', '--', ':']
    for i in range(0, len(n_states)):
        plt.title(title)
        plt.plot(x, y_allstates[i], label=n_states[i]+" states", \
            linestyle=line_styles[i], marker="o")
        plt.legend(loc=legendLoc, frameon=False)
    plt.savefig(f)
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
        "evaluation")
    if topology == "LR":
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.0005, 0.0005, 0.001]
    else:
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.003, 0.003, 0.003]
    save_boxplot(data=ev_list, title="Evaluation: probability error", \
        topology=topology, suffix="evaluation_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)


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
        "decoding")
    if topology == "LR":
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.001, 0.001, 0.001]
    else:
        tmp_ybot = [0.0,0.0,0.0]
        tmp_ytop = [0.001, 0.001, 0.001]
    save_boxplot(data=de_list, title="Decoding: path error increment", \
        topology=topology, suffix="decoding_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)


def summarize_training(filename):
    kl = []
    tr = []
    pi = []
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
    return kl, tr, pi


def plot_training(topology):
    tr_kl_list = []
    tr_tr_list = []
    tr_in_list = []
    y_kl_all = []
    y_tr_all = []
    y_in_all = []
    for n in n_states:
        tr_kl = []
        tr_tr = []
        tr_in = []
        y_kl = []
        y_tr = []
        y_in = []
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
            # read standard training results
            tr_kl_std, tr_tr_std, tr_in_std = summarize_training(f+f_tr_std)
            # read compressed training results
            tr_kl_compr, tr_tr_compr, tr_in_compr = summarize_training(f+f_tr_compr)
            for i in range(0, len(tr_kl_std)):
                tr_kl_diff.append(tr_kl_compr[i] - tr_kl_std[i])
            for i in range(0, len(tr_tr_std)):
                tr_tr_diff.append(tr_tr_compr[i] - tr_tr_std[i])
            for i in range(0, len(tr_in_std)):
                tr_in_diff.append(tr_in_compr[i] - tr_in_std[i])
            tr_kl.append(tr_kl_diff)
            tr_tr.append(tr_tr_diff)
            tr_in.append(tr_in_diff)
            y_kl.append(median(tr_kl_diff))
            y_tr.append(median(tr_tr_diff))
            y_in.append(median(tr_in_diff))
        tr_kl_list.append(tr_kl)
        tr_tr_list.append(tr_tr)
        tr_in_list.append(tr_in)
        y_kl_all.append(y_kl)
        y_tr_all.append(y_tr)
        y_in_all.append(y_in)

    save_plot(x, y_kl_all, "Training: KL-divergence difference", topology, "training_kl")
    save_plot(x, y_tr_all, "Training: Transitions relative error difference", topology, "training_tr")
    save_plot(x, y_in_all, "Training: Initial distribution error difference", topology, "training_in")

    # Training KL
    if topology == "CI":
        tmp_ybot = [-0.01, -0.5, -2]
        tmp_ytop = [0.01, 0.3, 0.5]
    elif topology == "FC":
        tmp_ybot = [-0.02, -1, -2]
        tmp_ytop = [0.01, 0.3, 1]
    elif topology == "LR":
        tmp_ybot = [-0.1, -8, -14]
        tmp_ytop = [0.05, 8, 10]
    save_boxplot(data=tr_kl_list, title="Training: states error", \
        topology=topology, suffix="training_kl_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)
    # Training TR
    tmp_ybot = [-0.5, -0.5, -0.5]
    tmp_ytop = [0.5, 0.5, 0.5]
    save_boxplot(data=tr_tr_list, title="Training: transitions error", \
        topology=topology, suffix="training_tr_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)
    # Training IN
    if topology == "LR":
        tmp_ybot = [-0.000011, -0.000011, -0.000011]
        tmp_ytop = [0.0000005, 0.0000005, 0.0000005]
    else:
        tmp_ybot = [-0.00000025, -0.000005, -0.00001]
        tmp_ytop = [0.00000005, 0.0000005, 0.0000005]
    save_boxplot(data=tr_in_list, title="Training: initial distribution error", \
        topology=topology, suffix="training_in_boxplot", ybot=tmp_ybot, \
        ytop=tmp_ytop)


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
                    floatlist_ratio.append(floatlist_std[i] / floatlist_compr[i])
                speedup.append(floatlist_ratio)
                y.append(median(floatlist_ratio))
            speedup_list.append(speedup)
            y_all.append(y)
        save_plot(x, y_all, "Speedup: "+p, topology, "speedup_"+p)
        if topology == "CI":
            if p == "evaluation":
                tmp_ybot = [0.0, 0.5, 1.0]
                tmp_ytop = [1.0, 1.5, 2.0]
            elif p == "decoding":
                tmp_ybot = [0.5, 0.5, 0.5]
                tmp_ytop = [1.5, 1.5, 1.5]
            elif p =="training":
                tmp_ybot = [0, 0, 0]
                tmp_ytop = [300, 600, 1500]
        elif topology == "FC":
            if p == "evaluation":
                tmp_ybot = [0.4, 0.75, 2]
                tmp_ytop = [0.9, 1.75, 4]
            elif p == "decoding":
                tmp_ybot = [0.5, 0.5, 0.75]
                tmp_ytop = [1.25, 1.25, 1.75]
            elif p == "training":
                tmp_ybot = [0, 0, 0]
                tmp_ytop = [300, 700, 1500]
        elif topology == "LR":
            if p == "evaluation":
                tmp_ybot = [0.25, 0.5, 1.25]
                tmp_ytop = [1, 1.25, 2.25]
            elif p == "decoding":
                tmp_ybot = [0.5, 0.75, 1]
                tmp_ytop = [1.25, 1.5, 2]
            elif p == "training":
                tmp_ybot = [0, 0, 0]
                tmp_ytop = [400, 700, 1700]
        save_boxplot(data=speedup_list, title="Speedup: "+p, \
            topology=topology, suffix="speedup_"+p+"_boxplot", ybot=tmp_ybot, \
            ytop=tmp_ytop)

if __name__ == "__main__":
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
            subprocess.call("convert "+folder+f+" "+folder+f[:-4]+"_boxplot.png +append " + \
                folder+"MERGE_"+f[5:], shell=True)
