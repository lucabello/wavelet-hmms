WAHMM_EVALUATION = ["--model", "data/model.yaml", "--obs",
                    "data/observations", "--binary", "--silence", "--tofile",
                    "--evaluation"]
WAHMM_DECODING = ["--model", "data/model.yaml", "--obs",
                  "data/observations", "--binary", "--silence", "--tofile",
                  "--decoding"]
WAHMM_TRAINING = ["--model", "data/model.yaml", "--obs",
                  "data/observations", "--binary", "--silence", "--tofile",
                  "--training"]
WAHMM_EVALUATION_COMPRESSED = ["--model", "data/model.yaml", "--obs",
                               "data/observations", "--binary", "--silence",
                               "--tofile", "--evaluation", "--compressed"]
WAHMM_DECODING_COMPRESSED = ["--model", "data/model.yaml", "--obs",
                             "data/observations", "--binary", "--silence",
                             "--tofile", "--decoding", "--compressed"]
WAHMM_TRAINING_COMPRESSED = ["--model", "data/model.yaml", "--obs",
                             "data/observations", "--binary", "--silence",
                             "--tofile", "--training", "--compressed"]

RESULTS_SKELETON = """---
executedTests: 0
evaluation:
  standard:
    probability: []
    time: []
  compressed:
    probability: []
    time: []
decoding:
  standard:
    probability: []
    error: []
    time: []
  compressed:
    probability: []
    error: []
    time: []
training:
  standard:
    kl: []
    logTransitionsError: []
    logInitialError: []
    iterations: []
    likelihood: []
    time: []
  compressed:
    kl: []
    logTransitionsError: []
    logInitialError: []
    iterations: []
    likelihood: []
    time: []
"""
