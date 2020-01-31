import pomegranate as pm
import numpy as np

# Example: create model with two states
model = pm.HiddenMarkovModel("Two-states HMM")
s1 = pm.State(pm.NormalDistribution(5, 1), name="State 1")
s2 = pm.State(pm.NormalDistribution(15, 1), name="State 2")
model.add_states([s1, s2])
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.5)
model.add_transition(s2, s1, 0.5)
model.add_transition(s1, s2, 0.5)
model.add_transition(s2, s2, 0.5)
model.bake()

# generate samples from the model
chosen_len = 10000
samples = model.sample(length=chosen_len, path=True, random_state=None)
observations = samples[0]
state_path = samples[1]

# problem 1 : maximum likelihood, forward algorithm
# To compare log probabilities, you can consider the ratio between them:
# specifically, with two states we can see that:
#     prob_1/prob_2 = exp(log(prob_1) - log(prob_2))
# forward_result = model.forward(observations)
# print(forward_result)

# problem 2 : Viterbi decoding, Viterbi algorithm
# compute the predicted_path using Viterbi algorithm and count errors
predicted_path = model.viterbi(observations)
error_count = 0
for i in range(1, chosen_len):
    if state_path[i] != predicted_path[1][i][1]:
        error_count += 1
# print(error_count)

# problem 3 : expectation maximization, baum-welch
predicted_model = pm.HiddenMarkovModel("Predicted HMM")
sp1 = pm.State(pm.NormalDistribution(0, 5), name="PState 1")
sp2 = pm.State(pm.NormalDistribution(30, 10), name="PState 2")
predicted_model.add_states([sp1, sp2])
predicted_model.add_transition(predicted_model.start, sp1, 1.0)
predicted_model.add_transition(sp1, sp1, 0.3)
predicted_model.add_transition(sp2, sp1, 0.7)
predicted_model.add_transition(sp1, sp2, 0.7)
predicted_model.add_transition(sp2, sp2, 0.3)
predicted_model.bake()

predicted_model.fit(list([np.array(observations)]))
# edges in json have the following format:
# - (start node, end node, probability, pseudocount, label)
# print(predicted_model.to_json())
