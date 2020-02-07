import pomegranate as pm
import numpy as np

# Example: create model with two states
model = pm.HiddenMarkovModel("Two-states HMM")
s1 = pm.State(pm.NormalDistribution(0, 1), name="State 1")
s2 = pm.State(pm.NormalDistribution(10, 1), name="State 2")
model.add_states([s1, s2])
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.99)
model.add_transition(s2, s1, 0.01)
model.add_transition(s1, s2, 0.01)
model.add_transition(s2, s2, 0.99)
model.bake()

# generate samples from the model
chosen_len = 100000
samples = model.sample(length=chosen_len, path=True, random_state=None)
observations = samples[0]
state_path = samples[1]

# print(observations)
out_file = open("data",'w')
for o in observations:
    out_file.write(str(o) + " ")
