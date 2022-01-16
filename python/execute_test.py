import yaml
import os
import shutil
import utilities.functions as functions
import utilities.io as io
import utilities.hmms as hmms
import utilities.kmeans as kmeans
import utilities.constants as constants


def execute_test(topology, state, eta, config):
    folder = functions.random_string()
    # Create subdirectories
    try:
        os.makedirs(folder)
    except Exception:
        print("[T="+topology+" S="+str(state)+" E="+str(eta)+" F="+folder+"]",
            "ERROR. The random folder name collides with an already existing",
            "one. Please rerun the script, as this should almost never happen.")
        return
    os.chdir(folder)
    os.makedirs("data")
    os.makedirs("data/kmeans")
    os.makedirs("results")
    results_file = os.path.join("../", topology + "_" + str(state) + "_"
                                + str(eta) + ".yaml")

    # Generate the necessary files for the test
    _prepareTest(topology, state, eta, config)
    # Perform the tests
    evaluation_standard, evaluation_compressed = _evaluation(config)
    decoding_standard, decoding_compressed = _decoding(config)
    training_standard, training_compressed = _training(config)
    # Save the results to file
    test_results = {}
    test_results["evaluation"] = {
        "standard": evaluation_standard, "compressed": evaluation_compressed}
    test_results["decoding"] = {
        "standard": decoding_standard, "compressed": decoding_compressed}
    test_results["training"] = {
        "standard": training_standard, "compressed": training_compressed}
    io.append_results(test_results, results_file)
    if not config["flags"]["keepTests"]:
        os.chdir("..")
        shdir.rmtree(folder)




def _prepareTest(topology, state, eta, config):
    # Generate HMM
    self_transition = 1 - (config["observations"]["transitionsPerState"]
                           / config["observations"]["sequenceLength"])
    hmm = hmms.generate_hmm(topology, state, eta, self_transition)
    io.save_yaml(hmm, "data/model.yaml")
    # Generate data
    observations, state_path = hmms.generate_data(hmm, config["observations"]["sequenceLength"])
    io.write_observations(observations, "data/observations")
    io.write_path(state_path, "data/path")
    # Create model estimations to serve as starting points for training
    for i in range(0, config["tests"]["trainingsPerTest"]):
        io.save_yaml(kmeans.estimate_model(observations, state),
                     "data/kmeans/estimateModel" + str(i) + ".yaml")


def _evaluation(config):
    wahmm_binary = config["paths"]["wahmmBinary"]
    # Run the algorithms
    standard_time = functions.execute_and_time(
        [wahmm_binary] + constants.WAHMM_EVALUATION)
    compressed_time = functions.execute_and_time(
        [wahmm_binary] + constants.WAHMM_EVALUATION_COMPRESSED)
    # Read results from files
    standard_probability = io.read_float("results/evaluation_probability")
    compressed_probability = io.read_float(
        "results/compressed_evaluation_probability")
    # Compute metrics
    standard_result = {}
    standard_result["probability"] = standard_probability
    standard_result["time"] = standard_time
    compressed_result = {}
    compressed_result["probability"] = compressed_probability
    compressed_result["time"] = compressed_time
    return standard_result, compressed_result


def _decoding(config):
    wahmm_binary = config["paths"]["wahmmBinary"]
    # Run the algorithms
    standard_time = functions.execute_and_time(
        [wahmm_binary] + constants.WAHMM_DECODING)
    compressed_time = functions.execute_and_time(
        [wahmm_binary] + constants.WAHMM_DECODING_COMPRESSED)
    # Read results from files
    standard_probability = io.read_float("results/decoding_probability")
    compressed_probability = io.read_float(
        "results/compressed_decoding_probability")
    standard_differences = functions.state_path_differences(
        "results/decoding_path", "data/path")
    compressed_differences = functions.state_path_differences(
        "results/compressed_decoding_path", "data/path")
    # Compute metrics
    standard_result = {}
    standard_result["probability"] = standard_probability
    standard_result["error"] = standard_differences / \
        config["observations"]["sequenceLength"]
    standard_result["time"] = standard_time
    compressed_result = {}
    compressed_result["probability"] = compressed_probability
    compressed_result["error"] = compressed_differences / \
        config["observations"]["sequenceLength"]
    compressed_result["time"] = compressed_time
    return standard_result, compressed_result


def _training(config):
    best_standard = {}
    best_standard["iterations"] = 101 # More than Baum-Welch maximum iterations
    best_standard["likelihood"] = 0
    best_compressed = best_standard.copy()
    for i in range(0, config["tests"]["trainingsPerTest"]):
        wahmm_binary = config["paths"]["wahmmBinary"]
        # Run the algorithms
        standard_time = functions.execute_and_time(
            [wahmm_binary] + constants.WAHMM_TRAINING + ["--estimate", "data/kmeans/estimateModel" + str(i) + ".yaml"])
        compressed_time = functions.execute_and_time(
            [wahmm_binary] + constants.WAHMM_TRAINING_COMPRESSED + ["--estimate", "data/kmeans/estimateModel" + str(i) + ".yaml"])
        # Read results from files
        real_model = yaml.safe_load("data/model")
        standard_model = yaml.safe_load("results/training_model")
        compressed_model = yaml.safe_load("results/compressed_training_model")
        # Check if this test is the best so far, and if so compute the metrics
        if (standard_model["iterations"] < best_standard["iterations"]) or \
            (standard_model["iterations"] == best_standard["iterations"] and \
            standard_model["likelihood"] > best_standard["likelihood"]):
            standard_model = hmms.reorder_states(standard_model, real_model)
            standard_kl_divergences = hmms.kl_divergences(real_model, standard_model)
            standard_transitions = hmms.transitions_error(standard_model, real_model)
            standard_initial = functions.compute_error(
            standard_model["logInitialDistribution"][0],
                real_model["logInitialDistribution"][0])
        # Do the same for compressed training
        if (compressed_model["iterations"] < best_compressed["iterations"]) or \
            (compressed_model["iterations"] == best_compressed["iterations"] and \
            compressed_model["likelihood"] > best_compressed["likelihood"]):
            compressed_model = hmms.reorder_states(compressed_model, real_model)
            compressed_kl_divergences = hmms.kl_divergences(
                real_model, compressed_model)
            compressed_transitions = hmms.transitions_error(
                compressed_model, real_model)
            compressed_initial = functions.compute_error(
                compressed_model["logInitialDistribution"][0],
                real_model["logInitialDistribution"][0])
    standard_result = {}
    standard_result["kl"] = standard_kl_divergences
    standard_result["logTransitionsError"] = standard_transitions
    standard_result["logInitialError"] = standard_initial
    standard_result["iterations"] = standard_model["iterations"]
    standard_result["likelihood"] = standard_model["likelihood"]
    standard_result["time"] = standard_time
    compressed_result = {}
    compressed_result["kl"] = compressed_kl_divergences
    compressed_result["logTransitionsError"] = compressed_transitions
    compressed_result["logInitialError"] = compressed_initial
    compressed_result["iterations"] = compressed_model["iterations"]
    compressed_result["likelihood"] = compressed_model["likelihood"]
    compressed_result["time"] = compressed_time
    return standard_result, compressed_result
