# import pomegranate as pm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt, log

# OPTIONS
use_binary_file = True
input_filename = "data/bin_observations"
output_filename = "data/kmeans_model"
labels_out_filename = "data/kmeans_labels"

# read observations from input file
in_file = open(input_filename, "r")
if use_binary_file == True:
    observations = np.fromfile(in_file, dtype=np.float64)
else:
    line = in_file.read()
    string_list = line.split()
    observations = []
    for s in string_list:
        observations.append(float(s))
in_file.close()

# cluster the observations to produce a starting model for baum-welch alg
k = 2
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

# write estimated model to file
out_file = open(output_filename, "w")
out_file.write(str(k) + " ") # number of states
for a in range(0, k): # states, from kmeans
    out_file.write(str(centroids[a][0]) + " " + str(std_devs[a]) + " ")
for a in range(0, k): # transitions, from uniform distribution
    for b in range(0, k):
        out_file.write(str(log(1/k)) + " ");
for a in range(0, k): # initial distributions, from uniform distribution
    out_file.write(str(log(1/k)) + " ");
out_file.close()

# write predicted labels for plotting purposes
out_file = open(labels_out_filename, "w")
for x in kmeans.labels_ :
    out_file.write(str(x) + " ") # number of states
out_file.close()
