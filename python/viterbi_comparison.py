#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import utilities_io as uio

# OPTIONS
verbose = False

def count_differences(p1, p2):
    difference_count = 0
    for i in range(0, len(p1)):
        if p1[i] != p2[i]:
            difference_count = difference_count + 1
            if verbose:
                print("index:",i,"p1:",p1[i],"p2:",p2[i])
    if verbose:
        print("Number of matches: ", len(p1)-difference_count)
        print("Number of differences: ", difference_count)
    return difference_count

def count_differences_uncompressed():
    state_path = uio.read_path()
    wahmm_path = uio.read_path("results/decoding_path")
    return count_differences(state_path, wahmm_path)

def count_differences_compressed():
    state_path = uio.read_path()
    compressed_path = uio.read_path("results/compressed_decoding_path")
    return count_differences(state_path, compressed_path)

if __name__ == "__main__":
    verbose = True
    # pm_path = uio.read_path("results/pm_decoding_path")
    wahmm_path = uio.read_path("results/decoding_path")
    compressed_path = uio.read_path("results/compressed_decoding_path")

    # print("--- PM decoding vs. WaHMM decoding ---")
    # count_differences(pm_path, wahmm_path)
    #
    # print("--- PM decoding vs. Compressed decoding ---")
    # count_differences(pm_path, compressed_path)

    print("--- WaHMM decoding vs. Compressed decoding ---")
    count_differences(wahmm_path, compressed_path)
