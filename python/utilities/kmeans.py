import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt, log, exp
import utilities.io

def estimate_model(observations, k):
    # Computer K-Means
    centroids, std_devs, labels = _kmeans(observations, k)
    # Sort the states by increasing mean
    sorted_centroids, sorted_devs, sorted_labels = _sortResults(centroids, std_devs, labels)
    # Build the estimated HMM with uniform probabilities
    hmm_kmeans = {}
    hmm_kemans["statesNumber"] = k
    hmm_kmeans["states"] = [[c[0],d] for (c,d) in zip(sorted_centroids, sorted_devs)]
    logTransitionRow = [log(1/k)]*k
    hmm_kmeans["logTransitions"] = [logTransitionRow]*k
    hmm_kmeans["logInitialDistribution"] = [log(1/k)]*k
    return hmm_kmeans


def _kmeans(observations, k):
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


def _sortResults(centroids, std_devs, labels):
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
