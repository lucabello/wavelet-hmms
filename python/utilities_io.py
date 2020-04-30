"""
Functions in this file:
def read_observations(f=observations_file, limit=0):
def read_observations_binary(f=bin_observations_file):
def read_path(f=state_path_file, limit=0):
def read_model(f=model_file):
def read_kmeans_model(f=kmeans_model_file):
"""
import numpy as np
from math import sqrt, log, exp

# common parameters
observations_file = "data/observations"
bin_observations_file = "data/bin_observations"
state_path_file = "data/path"
model_file = "data/model"
tests_folder = "tests/"

# Observations are a series of floats on a single line separated by spaces
def read_observations(f=observations_file, limit=0):
    in_file = open(f, "r")
    line = in_file.read()
    string_list = line.split()
    observations = []
    counter = 1
    for s in string_list:
        observations.append(float(s))
        counter = counter + 1
        if limit != 0 and counter > limit:
            break
    in_file.close()
    return observations

def write_observations(observations, f=observations_file):
    out_file = open(f, "w")
    obs_written = 1
    for o in observations:
        if obs_written == len(observations):
            out_file.write(str(o))
        else:
            out_file.write(str(o) + " ")
        obs_written = obs_written + 1
    out_file.close()

# Observations are the binary representation of a np.float64 array
def read_observations_binary(f=bin_observations_file):
    in_file = open(f, "rb")
    observations = np.fromfile(in_file, dtype=np.float64)
    return observations

def write_observations_binary(observations, f=bin_observations_file):
    out_file = open(f, "wb")
    np.array(observations, dtype=np.float64).tofile(out_file)
    out_file.close()

# Path is a series of ints on a single line separated by spaces
def read_path(f=state_path_file, limit=0):
    path_file = open(f, "r")
    line = path_file.read()
    string_list = line.split()
    path_list = []
    counter = 1
    for s in string_list:
        path_list.append(int(s))
        counter = counter + 1
        if limit != 0 and counter > limit:
            break
    path_file.close()
    return path_list

# state path param is from pomegranate model.sample()
def write_path(state_path, f=state_path_file):
    path_file = open(f, "w");
    for s in state_path:
        if s.name == ("None-start"): # initial silent state
            continue
        # strip the 's' from state name "s0", "s1", ...
        path_file.write(str(s.name)[-1:] + " ")
    path_file.close()

# The model is a series of numbers separated by spaces, following this format:
# n_states mean std_dev ... log(transitions) ... log(initial_distribution) ...
def read_model(f=model_file):
    in_file = open(f, "r")
    line = in_file.read()
    v = line.split()
    n_states = int(v[0])
    means = []
    std_devs = []
    offset = 1
    for i in range(0, n_states):
        means.append(float(v[offset + 2*i]))
        std_devs.append(float(v[offset + 2*i + 1]))
    transitions = []
    offset = 1 + n_states*2
    for i in range(0, n_states):
        for j in range(0, n_states):
            transitions.append(exp(float(v[offset])))
            offset = offset + 1
    initial = []
    for i in range(0, n_states):
        initial.append(exp(float(v[offset])))
        offset = offset + 1
    in_file.close()
    return n_states, means, std_devs, transitions, initial

# Write list to a file
def write_list(f, l):
    out_file = open(f, "w")
    for x in l:
        out_file.write(str(x)+" ")
    out_file.close()
