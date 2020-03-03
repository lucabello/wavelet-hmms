import pomegranate as pm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt, log

# OPTIONS
sequence_length = 10000
save_model = True
perform_kmeans = True

# create model with two states
model = pm.HiddenMarkovModel("Two-states HMM")
s1 = pm.State(pm.NormalDistribution(0, 1), name="0")
s2 = pm.State(pm.NormalDistribution(10, 1), name="1")
model.add_states([s1, s2])
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.5)
model.add_transition(s2, s1, 0.5)
model.add_transition(s1, s2, 0.5)
model.add_transition(s2, s2, 0.5)
model.bake()

# save the model to file
if save_model == True:
    out_file = open("data/model", "w")
    out_file.write("2 0 1 10 1 ")
    out_file.write(str(log(0.5)) + " " + str(log(0.5)) + " ")
    out_file.write(str(log(0.5)) + " " + str(log(0.5)) + " ")
    out_file.write(str(log(1)) + " " + "-inf ")
    out_file.close()

# generate samples from the model
samples = model.sample(length=sequence_length, path=True, random_state=None)
observations = samples[0]
state_path = samples[1]

# cluster the observations to produce a starting model for baum-welch alg
if perform_kmeans == True:
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
    out_file = open("data/kmeans_model", "w")
    out_file.write(str(k) + " ") # number of states
    for a in range(0, k): # states, from kmeans
        out_file.write(str(centroids[a][0]) + " " + str(std_devs[a]) + " ")
    for a in range(0, k): # transitions, from uniform distribution
        for b in range(0, k):
            out_file.write(str(log(1/k)) + " ");
    for a in range(0, k): # initial distributions, from uniform distribution
        out_file.write(str(log(1/k)) + " ");
    out_file.close()

# save readable data
out_file = open("data/observations", "w")
obs_written = 1
for o in observations:
    if obs_written == len(observations):
        out_file.write(str(o))
    else:
        out_file.write(str(o) + " ")
    obs_written = obs_written + 1
out_file.close()
# save readable path
path_file = open("data/path", "w");
for s in state_path:
    if s.name == ("Two-states HMM-start"):
        continue
    path_file.write(str(s.name) + " ")
path_file.close()
