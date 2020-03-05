#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from math import exp
import utilities_io as uio
import utilities_kmeans as ukm

# read kmeans model from input file
n_states, means, std_devs, transitions, initial = ukm.read_kmeans_model()

# read kmeans labels from file
labels = ukm.read_kmeans_labels()

# read observations from input file
in_file = open(uio.observations_file, "r")
line = in_file.read()
string_list = line.split()
value_list = []
value_x = []
for i in range(0, n_states):
    value_list.append([])
    value_x.append([])
input_limit = 1000000
counter = 0
for s in string_list:
    value_list[labels[counter]].append(float(s))
    value_x[labels[counter]].append(counter)
    counter = counter + 1
    if counter > input_limit:
        break
in_file.close()

for i in range(0, n_states):
    print("State",i,"- Mean:",means[i],"- StdDev:",std_devs[i])

x = range(1, counter+1)
means_plot = []
std_low_plot = []
std_high_plot = []
for i in range(0, n_states):
    means_plot.append([means[i]] * len(x))
    std_low_plot.append([means[i] - 3*std_devs[i]] * len(x))
    std_high_plot.append([means[i] + 3*std_devs[i]] * len(x))

# plt.scatter(x, value_list, color='blue')
for i in range(0, n_states):
    # -- either differentiate clusters
    plt.scatter(value_x[i], value_list[i])
    # -- or just print observations
    # plt.scatter(value_x[i], value_list[i], color='blue')

    plt.plot(x, means_plot[i], color='black')
    # plt.plot(x, std_low_plot[i], color='red')
    # plt.plot(x, std_high_plot[i], color='red')
plt.show()
