import pomegranate as pm
import numpy as np
import networkx as nx
import utilities.functions as functions
from math import log, exp


def generate_hmm(topology, state, eta, self_transition):
    hmm = {}
    hmm["statesNumber"] = state
    # Generate states using eta, all with unitary standard deviation
    generated_states = [[0.0, 1.0]]
    for i in range(1, state):
        generated_states.append([6*eta + generated_states[i-1][0], 1.0])
    # hmm["states"] = [str(gs) for gs in generated_states] //TODO: remove line
    hmm["states"] = generated_states
    hmm["logTransitions"] = _generate_transitions(topology, state, self_transition)
    generated_initial_distribution = ["-inf" for i in range(1, state)]
    generated_initial_distribution.insert(0, log(1))
    hmm["logInitialDistribution"] = generated_initial_distribution
    return hmm


def _generate_transitions(topology, state, self_transition):
    transitions = [[0]*state]*state
    if topology == "FullyConnected":
        transition_out = (1 - self_transition) / (state - 1)
        for i in range(0, state):
            for j in range(0, state):
                if i == j:  # Self-transition
                    transitions[i][j] = log(self_transition)
                else:  # Transition out
                    transitions[i][j] = log(transition_out)
    if topology == "Circular":
        transition_out = 1 - self_transition
        for i in range(0, state):
            for j in range(0, state):
                if j == 0 and i == state-1:  # Last state
                    transitions[i][j] = log(transition_out)
                elif i == j:  # Self-transition
                    transitions[i][j] = log(self_transition)
                elif j == i+1:  # Transition out
                    transitions[i][j] = log(transition_out)
                else:
                    transitions[i][j] = "-inf"
    if topology == "LeftToRight":
        transition_out = 1 - self_transition
        for i in range(0, state):
            for j in range(0, state):
                if i == j and i == state-1:  # Last state is absorbing
                    transitions[i][j] = log(1)
                elif i == j:  # Self-transition
                    transitions[i][j] = log(self_transition)
                elif i == j-1:  # Transition out
                    transitions[i][j] = log(transition_out)
                else:
                    transitions[i][j] = "-inf"
    return transitions


def generate_data(hmm, sequence_length):
    states_number = hmm["statesNumber"]
    # Create Pomegranate model
    dists = []
    for i in range(0, states_number):
        dists.append(pm.NormalDistribution(float(hmm["states"][i][0]),
                                           float(hmm["states"][i][1])))
    trans_mat = []
    for i in range(0, states_number):
        trans_mat.append([])
        for j in range(0, states_number):
            trans_mat[i].append(exp(float(hmm["logTransitions"][i][j])))
    trans_mat = np.array(trans_mat)
    starts = np.array([exp(float(init)) for init in hmm["logInitialDistribution"]])
    # starts = np.array(hmm["logInitialDistribution"])
    model = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts)
    model.bake()

    # Generate observations and state path from the model
    samples = model.sample(length=sequence_length,
                           path=True, random_state=None)
    observations = samples[0]
    state_path = samples[1]

    # Save to file
    return observations, state_path


def reorder_states(trained_model, base_model):
    states_number = base_model["statesNumber"]
    # Find max KL-divergence
    max_kl = -1
    for i in range(0, states_number):
        for j in range(0, states_number):
            kl = functions._compute_kl(base_model["states"][i],
                                       trained_model["states"][i])
            if kl > max_kl:
                max_kl = kl
    # Setup NX Graph to reorder the states with max_weight_matching
    graph = nx.Graph()
    for i in range(0, states_number):
        graph.add_node("base"+str(i))
        graph.add_node("trained"+str(i))
    for i in range(0, states_number):
        for j in range(0, states_number):
            graph.add_edge("base"+str(i), "trained"+str(j),
                           weight=(max_kl-functions._compute_kl(
                               base_model["states"][i],
                               trained_model["states"][j])))
    mates = dict(nx.max_weight_matching(graph, maxcardinality=True))
    new_order = [-1] * states_number
    for v in mates:
        if "base" in v:
            new_order[int(v[1:])] = int(mates[v][1:])
        else:
            new_order[int(mates[v][1:])] = int(v[1:])
    reordered_model = trained_model
    for i in range(0, states_number):
        reordered_model["states"][i] = reordered_model["states"][new_order[i]]
        reordered_model["logTransitions"][i] = \
            reordered_model["logTransitions"][new_order[i]]
        reordered_model["logInitialDistribution"][i] = \
            reordered_model["logInitialDistribution"][new_order[i]]
    return reordered_model


def kl_divergences(model1, model2):
    divergences = []
    for i in range(0, model1["statesNumber"]):
        kl = functions.compute_kl(model1["states"][i], model2["states"][i])
        divergences.append(kl)
    return divergences


def transitions_error(trained_model, base_model):
    states_number = base_model["statesNumber"]
    errors = [[0]*states_number]*states_number
    for i in range(0, states_number):
        for j in range(0, states_number):
            errors[i][j] = functions.compute_error(
                trained_model["logTransitions"][i][j],
                base_model["logTransitions"][i][j])
    return errors
