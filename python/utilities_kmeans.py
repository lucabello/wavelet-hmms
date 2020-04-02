"""
Functions in this file:
def read_observations(f=observations_file, limit=0):
def read_observations_binary(f=bin_observations_file):
def read_path(f=state_path_file, limit=0):
def read_model(f=model_file):
def read_kmeans_model(f=kmeans_model_file):
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt, log, exp
import utilities_io as uio

kmeans_model_file = "data/kmeans_model"
kmeans_labels_file = "data/kmeans_labels"

def read_kmeans_model(f=kmeans_model_file):
    return uio.read_model(f)

def write_kmeans_model(centroids, std_devs, f=kmeans_model_file):
    out_file = open(f, "w")
    k = len(centroids)
    out_file.write(str(k) + " ") # number of states
    for a in range(0, k): # states, from kmeans
        out_file.write(str(centroids[a][0]) + " " + str(std_devs[a]) + " ")
    for a in range(0, k): # transitions, from uniform distribution
        for b in range(0, k):
            out_file.write(str(log(1/k)) + " ");
    for a in range(0, k): # initial distributions, from uniform distribution
        out_file.write(str(log(1/k)) + " ");
    out_file.close()

def read_kmeans_labels(f=kmeans_labels_file):
    in_file = open(f, "r")
    line = in_file.read()
    string_list = line.split()
    labels_list = []
    for s in string_list:
        labels_list.append(int(s))
    return labels_list

def write_kmeans_labels(labels, f=kmeans_labels_file):
    out_file = open(f, "w")
    for x in labels:
        out_file.write(str(x) + " ") # number of states
    out_file.close()

def kmeans(observations, k):
    df = pd.DataFrame(observations)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_ # means
    # calculate the standard deviations
    o_count = 0
    std_devs = [0] * k
    state_counts = [0] * k
    for o in observations:
        i = kmeans.labels_[o_count] # state of the current observation
        std_devs[i] = std_devs[i] + (o - centroids[i])**2
        state_counts[i] = state_counts[i] + 1
        o_count = o_count + 1
    for a in range(0, k):
        std_devs[a] = sqrt(std_devs[a]/state_counts[a])
    labels = kmeans.labels_
    return centroids, std_devs, labels

def estimate_model(k, use_binary_file=True):
    if use_binary_file == True:
        observations = uio.read_observations_binary()
    else:
        observations = uio.read_observations()
    centroids, std_devs, labels = kmeans(observations, k)
    # sort states by increasing mean
    initial_order = []
    new_order = []
    for i in range(0, len(centroids)):
        initial_order.append(i)
        new_order.append(i)
    for i in range(0, len(centroids)):
        for j in range(1, len(centroids)-i):
            if centroids[j-1] > centroids[j]:
                tmp = new_order[j-1]
                new_order[j-1] = new_order[j]
                new_order[j] = tmp
    sorted_centroids = []
    sorted_devs = []
    for i in range(0, len(centroids)):
        sorted_centroids.append(centroids[new_order[i]])
        sorted_devs.append(std_devs[new_order[i]])
    sorted_labels = []
    for i in range(0, len(labels)):
        sorted_labels.append(new_order[labels[i]])

    write_kmeans_model(sorted_centroids, sorted_devs)
    write_kmeans_labels(sorted_labels)
