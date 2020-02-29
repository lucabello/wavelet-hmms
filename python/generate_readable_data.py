import pomegranate as pm
import numpy as np

# Example: create model with two states
model = pm.HiddenMarkovModel("Two-states HMM")
s1 = pm.State(pm.NormalDistribution(0, 1), name="0")
s2 = pm.State(pm.NormalDistribution(20, 1), name="1")
model.add_states([s1, s2])
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s1, 0.9999)
model.add_transition(s2, s1, 0.0001)
model.add_transition(s1, s2, 0.0001)
model.add_transition(s2, s2, 0.9999)
model.bake()

# generate samples from the model
chosen_len = 100000000
samples = model.sample(length=chosen_len, path=True, random_state=None)
observations = samples[0]
state_path = samples[1]

# save readable data
out_file = open("data/observations", "w")
obs_written = 1
for o in observations:
    if obs_written == len(observations):
        out_file.write(str(o))
    else:
        out_file.write(str(o) + " ")
    obs_written = obs_written + 1
out_file.close()
# save readable path
path_file = open("data/path", "w");
for s in state_path:
    if s.name == ("Two-states HMM-start"):
        continue
    path_file.write(str(s.name) + " ")
path_file.close()
