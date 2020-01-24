# wavelet-hmms
## Machine learning for big-sequence data: Wavelet-compressed Hidden Markov Models

This is the repository holding the code for my master thesis. 
The following serves as a list of steps to execute and accomplish.

### Define different HMMs
Using a certain framework (e.g. gHMM), define different models. The parameters of interest will be two:
* the self-transition probability, that impacts how much you can compress data; this should be expressed as the number fo times we stay in the state in the model definition, to avoid underflow errors on other transition probabilities;
* the separation of the states, in terms of separation of means and small enough variance; the means generation is the solution the problem "find *n* random means, with one being 0, having at least ε separation"; the separation parameter ε can easily be a function of the variance.
In a HMM, the states are some numerical values, specifically the mean of the normal distribution; the emission pdf for a state is Normal with a certain constant variance.

### Generate data
Given a certain HMM that was previously defined, generate data saving both the observations and the generating path (the states). Just select a starting state and run the Markov chain for a certain number of steps.

### Apply standard algorithms
A standard version of the algorithms probably needs to be developed. There should be a big amount of material online to figure out how to do that, plus Rabiner's paper.
Likelihood computation (problem 1) computes the probability that a given sequence has been generated by a certain model; since that is the case, the probability should be pretty high.
Viterbi decoding removes the states from data and tries to infer them; the result should be a sequence as close as possible to the actual states.
Expectation maximization removes the model and tries to infer it from the observations and the states; the result should be a model as close as possible to the actual one.
At this point it should be verified that the implementation of the standard algorithms works by comparing the obtained results with the actual data.

### Apply compressed algorithms
First, compress the data using wavelets (specifically HaMMLET), thus obtaining the breakpoint array data structure. Then use the new algorithms on the compressed version of the data, comparing the results both with the uncompressed version and with the original data.

### Evaluate results
Results evaluation should be in terms of speed and accuracy, tweaking parameters in HMMs definitions to see how results change. The expectation is to have improvements when self-loop probabilities are high (a lot of time is spent in the same state) and the states are well separated.
