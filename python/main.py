import yaml
import sys
import itertools
import os
import threading
import utilities.constants as constants
from execute_test import execute_test
import utilities.io # Import to initialize Lock


# Check that the script is invoked correctly
if "-h" in sys.argv or "--help" in sys.argv:
    print("WaHMM automated test")
    print("automated_test_parallel.py <configuration_file>")
    exit(0)

if len(sys.argv) != 2:
    print("Error. Execute as: automated_test_parallel.py \
          < configuration_file >")
    exit(1)

# Read configuration from file
with open(sys.argv[1]) as file:
    config = yaml.safe_load(file)

topologies = config["hmm"]["topologies"]
states = config["hmm"]["statesNumber"]
etas = config["hmm"]["stateSeparation"]

for topology, state, eta in itertools.product(topologies, states, etas):
    # Define some paths from configuration
    tests_root = config["paths"]["testsRoot"]
    testset_id = topology + "_" + str(state) + "_" + str(eta)
    testset_root = os.path.join(tests_root, testset_id)
    # Create directories for the test set
    os.makedirs(tests_root, exist_ok=True)
    os.makedirs(testset_root, exist_ok=True)
    # Run the test set
    os.chdir(testset_root)
    # Check how many tests have already been executed
    executed_tests = 0
    results_file = testset_id + ".yaml"
    if os.path.exists(results_file):
        with open(results_file, "r") as file:
            results = yaml.safe_load(file)
            executed_tests = results["executedTests"]
    else:
        with open(results_file, "w+") as file:
            file.write(constants.RESULTS_SKELETON)
    # Execute the tests
    threads = []
    for test_counter in range(executed_tests, config["tests"]["testsNumber"]):
        if config["flags"]["multithread"]:
            th = threading.Thread(target=execute_test, args=(
                topology, state, eta, config,))
            th.start()
            threads.append(th)
        else:
            execute_test(topology=topology, state=state,
                         eta=eta, config=config)

    # Wait for threads to terminate before moving on to the next test set
    if config["flags"]["multithread"]:
        for th in threads:
            th.join()
    os.chdir("..")
