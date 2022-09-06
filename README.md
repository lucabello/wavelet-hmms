# WaHMM - Wavelet-compressed Hidden Markov Models

The version of the code used in the thesis and papers it's tagged under [v1.0-thesis](https://github.com/lucabello/wavelet-hmms/tree/v1.0-thesis). Please still follow the installation instructions reported here.

---

This is the respository holding the code for my master thesis written as en Erasmus student at Chalmers University (Gothenburg, Sweden). The thesis will be published by Chalmers and can be found at the following link:

[insert link]

## Abstract
Hidden Markov models are among the most important machine learning methods for the statistical analysis of sequential data, but they struggle when applied to big data. Their relative inefficiency has been addressed several times by the use of some compression techniques, either for the computation or for the data. This thesis explores the latter, with the application of a data compression technique based on wavelets and the subsequent adaptation of the main HMMs algorithms from the literature: the forward, Viterbi and Baum-Welch algorithms used to solve the evaluation, decoding and training problem respectively. The testing phase shows that this new technique generally yields equal or better results, obtaining some extremely high speedups in the training problem, making it even thousands of times faster; this enables the training of a HMM with big data on a commodity laptop.


## Installation

### Dependencies
Download these libraries and extract them into the `lib/` folder:
```
# HaMMLET
http://wiedenhoeft.github.io/HaMMLET/
# aixlog
https://github.com/badaix/aixlog
# cxxopts
https://github.com/jarro2783/cxxopts
```
