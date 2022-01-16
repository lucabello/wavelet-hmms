import numpy as np
import yaml
import os
import threading
import json


_lock = threading.Lock()


def save_yaml(dictionary, filename):
    with open(filename, "w") as file:
        yaml.dump(dictionary, file)


def read_float(filename):
    with open("filename", "r") as file:
        result = float(file.read())
    return result


def write_observations(observations, filename, binary=True):
    if binary:
        with open(filename, "wb") as out:
            np.array(observations, dtype=np.float64).tofile(out)
    else:
        with open(filename, "w") as out:
            for o in observations[:-1]:
                out.write(str(o) + " ")
            out.write(str(observations[-1]))


def write_path(state_path, filename):
    with open(filename, "w") as out:
        for s in state_path:
            if s.name == ("None-start"):  # Initial silent state
                continue
            # Strip the 's' from state name "s0", "s1", ...
            out.write(str(s.name)[-1:] + " ")


def append_results(test_results, filename):
    with _lock:
        results = yaml.safe_load(filename)
        results["executedTests"] += 1
        for mode in ["standard", "compressed"]:
            # Evaluation results
            results["evaluation"][mode]["probability"].append(
                test_results["evaluation"][mode]["probability"])
            results["evaluation"][mode]["time"].append(
                test_results["evaluation"][mode]["time"])
            # Decoding results
            results["decoding"][mode]["probability"].append(
                test_results["decoding"][mode]["probability"])
            results["decoding"][mode]["error"].append(
                test_results["decoding"][mode]["error"])
            results["decoding"][mode]["time"].append(
                test_results["decoding"][mode]["time"])
            # Training results
            results["training"][mode]["kl"].append(
                json.dumps(test_results["training"][mode]["kl"]))
            results["training"][mode]["logTransitionsError"].append(
                json.dumps(
                    test_results["training"][mode]["logTransitionsError"]))
            results["training"][mode]["logInitialError"].append(
                json.dumps(
                    test_results["training"][mode]["logTransitionsError"]))
            results["training"][mode]["iterations"].append(
                test_results["training"][mode]["iterations"])
            results["training"][mode]["likelihood"].append(
                test_results["training"][mode]["likelihood"])
            results["training"][mode]["time"].append(
                test_results["training"][mode]["time"])
        # Atomically write the file
        tmp_file = filename = "_tmp"
        save_yaml(results, tmp_file)
        os.replace(tmp_file, filename)
