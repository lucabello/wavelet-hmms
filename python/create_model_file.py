#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import log
import utilities_io as uio

# Creates a model file; the input params are inserted by hand

# The file should be written with the following format
# n_states mean std_dev ... log(transitions) ... log(initial_distribution) ...

out_file = open(uio.model_file, "w")

# number of states
out_file.write("2 ")
# states distributions
out_file.write("0 1 10 1 ")
# transition matrix
out_file.write(str(log(0.999)) + " " + str(log(0.001)) + " ")
out_file.write(str(log(0.001)) + " " + str(log(0.999)) + " ")
# initial distribution
out_file.write(str(log(1)) + " " + "-inf ")

out_file.close()
