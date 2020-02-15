import pomegranate as pm
import numpy as np

# Example: create model with two states
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

observations = []
# read observations
in_file = open("data/observations", "r")
for line in in_file:
    for o in line.split(' '):
        num = float(o)
        observations.append(num)

# problem 1 : maximum likelihood, forward algorithm
# To compare log probabilities, you can consider the ratio between them:
# specifically, with two states we can see that:
#     prob_1/prob_2 = exp(log(prob_1) - log(prob_2))
print("----- Evaluation Problem -----")
forward_result = model.forward(observations)
print(forward_result)
evaluation_result = model.log_probability(observations)
print("P(O|lambda):", evaluation_result)

# # problem 2 : Viterbi decoding, Viterbi algorithm
# # compute the predicted_path using Viterbi algorithm and count errors
# predicted_path = model.viterbi(observations)
# print(predicted_path)
# error_count = 0
# for i in range(1, chosen_len):
#     if state_path[i] != predicted_path[1][i][1]:
#         error_count += 1
# # print(error_count)
print("----- Decoding Problem -----")
logp_path, predicted_path = model.viterbi(observations)
predicted_path = model.predict(observations, "viterbi")
pathlen = len(predicted_path)-1 # don't count the silent starting state
# only makes sense if pathlen >= 10
print(predicted_path[1:6], "...", predicted_path[pathlen-4:])
# print(predicted_path)
print("P(Q|O,lambda):",logp_path)
viterbi_file = open("results/pomegranate_viterbi","w")
for p in predicted_path[1:]:
    viterbi_file.write(str(p) + " ")
viterbi_file.close()

# # problem 3 : expectation maximization, baum-welch
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
print(predicted_model.to_json())
