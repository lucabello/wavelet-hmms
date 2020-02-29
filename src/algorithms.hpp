#ifndef WAHMM_ALGORITHMS_HPP
#define WAHMM_ALGORITHMS_HPP
#include "commons.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

void evaluation_problem(Model& m, std::vector<wahmm::real_t>& obs, bool verbose);
wahmm::real_t** forward_matrix(Model& m, std::vector<wahmm::real_t>& obs);
wahmm::real_t** backward_matrix(Model& m, std::vector<wahmm::real_t>& obs);
void decoding_problem(Model &m, std::vector<wahmm::real_t>& obs, bool verbose);
void training_problem_wrapper(Model& m, std::vector<wahmm::real_t>& obs, wahmm::real_t thresh,
    size_t maxIterations, bool verbose);
/**
* Solve the evaluation problem through the forward algorithm.
* If verbose is true, print the results.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true prints result informations
*/
void evaluation_problem(Model& m, std::vector<wahmm::real_t>& obs, bool verbose){
    wahmm::real_t **logForward;
    wahmm::real_t logEvaluation;
    size_t numberOfStates = m.mStates.size();

    if(verbose)
        std::cout << "+++++ Evaluation Problem +++++" << std::endl;

    logForward = forward_matrix(m, obs); // initialization and induction
    // termination
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][obs.size()-1]);
    }

    // print results
    if(verbose){
        printMatrixSummary(logForward, numberOfStates, obs.size(),
            "logForward", false);
        std::cout << "log[ P(O|lambda) ] = " << logEvaluation << std::endl;
    }

    freeMatrix(logForward, numberOfStates);
}

/**
* Compute the forward matrix given a model and an observations sequence.
*
* @param m the model
* @param obs the observation sequence
*/
wahmm::real_t** forward_matrix(Model& m, std::vector<wahmm::real_t>& obs){
    wahmm::real_t **logForward;
    size_t numberOfStates = m.mStates.size();
    logForward = new wahmm::real_t*[numberOfStates]; // forward variables

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logForward[i] = new wahmm::real_t[obs.size()];
        // alpha_0(i) = pi_i * b_i(O_1)
        logForward[i][0] = m.mLogPi[i] +
            m.mStates[i].logPdf(obs[0]);
    }
    // induction
    for(size_t t = 1; t < obs.size(); t++){ // observations
        for(size_t j = 0; j < numberOfStates; j++){ // arriving state
            logForward[j][t] = -infin;
            for(int i = 0; i < numberOfStates; i++){ // starting state
                // alpha_t+1(j) = sum_{i=0}^N alpha_t(i)a_{ij} ...
                logForward[j][t] = sum_logarithms(logForward[j][t],
                    logForward[i][t-1] + m.mLogTransitions[i][j]);
            }
            // ... b_{j}(O_{t+1})
            logForward[j][t] += m.mStates[j].logPdf(obs[t]);
        }
    }

    return logForward;
}

/**
* Compute the backward matrix given a model and an observations sequence.
*
* @param m the model
* @param obs the observation sequence
*/
wahmm::real_t** backward_matrix(Model& m, std::vector<wahmm::real_t>& obs){
    wahmm::real_t **logBackward;
    size_t numberOfStates = m.mStates.size();
    logBackward = new wahmm::real_t*[numberOfStates]; // forward variables

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logBackward[i] = new wahmm::real_t[obs.size()];
        // beta_T(i) = 1
        logBackward[i][obs.size()-1] = 0;
    }
    // induction
    for(int t = obs.size()-2; t >= 0; t--){ // observations
        for(size_t i = 0; i < numberOfStates; i++){ // arriving state
            logBackward[i][t] = -infin;
            for(size_t j = 0; j < numberOfStates; j++){ // starting state
                // beta_t(i) = sum_{j=1}^N a_{ij} b_j(O_{t+1}) beta_{t+1}(j)
                logBackward[i][t] = sum_logarithms(logBackward[i][t],
                    m.mLogTransitions[i][j] +
                    m.mStates[j].logPdf(obs[t+1]) +
                    logBackward[j][t+1]);
            }
        }
    }

    return logBackward;
}


/**
* Solve the decoding problem through the forward algorithm.
* Currently write the obtained path in results/wahmm_viterbi.
* If verbose is true print the Viterbi matrix.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true print result informations
*/
void decoding_problem(Model &m, std::vector<wahmm::real_t>& obs, bool verbose){
    wahmm::real_t **logViterbi, **statesViterbi;
    size_t numberOfStates = m.mStates.size();
    logViterbi = new wahmm::real_t*[numberOfStates];
    statesViterbi = new wahmm::real_t*[numberOfStates];

    if(verbose)
        std::cout << "+++++ Decoding Problem +++++" << std::endl;

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logViterbi[i] = new wahmm::real_t[obs.size()];
        statesViterbi[i] = new wahmm::real_t[obs.size()];
        // delta_0(i) = pi_i * b_i(O_1)
        logViterbi[i][0] = m.mLogPi[i] + m.mStates[i].logPdf(obs[0]);
        statesViterbi[i][0] = -1; // psi_0(i) = 0
    }
    // induction
    wahmm::real_t currentMax = -infin;
    wahmm::real_t currentSum = 0;
    int currentState = -1;
    for(size_t t = 1; t < obs.size(); t++){ // observations
        for(size_t j = 0; j < numberOfStates; j++){ // arriving state
            logViterbi[j][t] = -infin;
            // delta_t(j) = max_{1<=i<=N} delta_{t-1}(i)a_{ij} ...
            for(size_t i = 0; i < numberOfStates; i++){ // starting state
                currentSum = logViterbi[i][t-1] + m.mLogTransitions[i][j];
                if(currentSum > currentMax){
                    currentMax = currentSum;
                    currentState = i;
                }
            }
            // ... b_{j}(O_{t})
            logViterbi[j][t] = currentMax + m.mStates[j].logPdf(obs[t]);
            // psi_t(j) = argmax[...]
            statesViterbi[j][t] = currentState;
            // re-initialize max variables for next loop
            currentMax = -infin;
            currentState = -1;
        }
    }

    // termination
    wahmm::real_t logDecoding;
    list<size_t> viterbiPath;
    currentMax = -infin; // this will contain the log probability of the path
    currentState = -1;
    for(size_t i = 0; i < numberOfStates; i++){
        if(logViterbi[i][obs.size()-1] > currentMax){
            currentMax = logViterbi[i][obs.size()-1];
            currentState = i;
        }
    }
    viterbiPath.push_front(currentState);
    for(int t = obs.size()-2; t >= 0; t--){
        // if currentState == -1, impossible path? can it happen?
        if(currentState >= 0)
            currentState = statesViterbi[currentState][t+1];
        else
            std::cerr << "[Error] Impossible Viterbi path!" << std::endl;
        viterbiPath.push_front(currentState);
    }

    // print results
    if(verbose){
        printMatrixSummary(logViterbi, numberOfStates, obs.size(),
            "logViterbi", false);
        std::cout << "Most likely path: " << std::endl;
        int i = 0;
        for(auto it = viterbiPath.begin(); it != viterbiPath.end(); it++, i++){
            // only print first 5 and last 5 states
            if(i == 5)
                std::cout << "... ";
            if(i >= 5 && i < viterbiPath.size() - 5)
                continue;
            std::cout << *it << " ";
        }
        std::cout << std::endl;
        std::cout << "log[ P(Q|O,lambda) ] = " << currentMax << std::endl;
    }

    // print to file for comparison with other implementations
    std::ofstream ofs ("results/wahmm_viterbi", std::ofstream::out);
    for(auto it = viterbiPath.begin(); it != viterbiPath.end(); it++)
        ofs << *it << " ";
    ofs.close();

    freeMatrix(logViterbi, numberOfStates);
    freeMatrix(statesViterbi, numberOfStates);
}

/**
* Solve the training problem performing one iteration of the Baum-Welch
* algorithm.
* The forward matrix is calculated entirely; the other things are not to save
* space; the backward variable is computed only for the current time; for the
* reestimation parameters, accumulate sufficient statistics at each time step.
* For example, the current gamma_t is accumulated into logGamma and also used
* to calculate the values for the other accumulators logAverage and logVariance.
* logEpsilon will contain, at the end, the new transition matrix.
* The P(O|lambda) cancels out in everything except for the initial distribution,
* so it's simplified and not included at all.
* For the backward variable, two arrays are needed, for beta_t and beta_{t+1}.
* For this reason, use two arrays and swap them at each time step.

// the reestimation of the average requires calculating log(obs[t]);
// to avoid negative observations, translate the sequence (only when
// performing that computation) so the minimum obsevation is zero.
//
// Actually, translate so that the sequence ranges from [1, inf) because
// zero could be represented as -0.0000000000001 and it would make the log
// function crash.

@returns the logEvaluation P(O|lambda) of the previous model

*/
wahmm::real_t training_problem(Model& m, std::vector<wahmm::real_t>& obs, wahmm::real_t minObs,
    wahmm::real_t **logEpsilon, wahmm::real_t *logBackward, wahmm::real_t *prevLogBackward,
    wahmm::real_t *logPi, wahmm::real_t **logGamma, wahmm::real_t *logGammaSum,
    wahmm::real_t *logAverage, wahmm::real_t *logVariance, bool verbose){

    wahmm::real_t logEvaluation; // P(O|lambda)
    wahmm::real_t **logForward; // forward matrix
    wahmm::real_t *tmp; // for swapping of the backward arrays
    size_t numberOfStates = m.mStates.size();

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        for(size_t j = 0; j < numberOfStates; j++)
            logEpsilon[i][j] = -infin;
        logBackward[i] = 0;
        prevLogBackward[i] = 0;
        logGammaSum[i] = -infin;
        logAverage[i] = -infin;
        logVariance[i] = -infin;
    }
    logForward = forward_matrix(m, obs);
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][obs.size()-1]);
    }

    // start from T-1
    for(int t = obs.size()-2; t >= 0; t--){
        // calculate backward variable for the next iteration
        for(int i = 0; i < numberOfStates; i++){
            logBackward[i] = -infin;
            for(size_t j = 0; j < numberOfStates; j++){
                logBackward[i] = sum_logarithms(logBackward[i],
                    m.mLogTransitions[i][j] +
                    m.mStates[j].logPdf(obs[t+1]) +
                    prevLogBackward[j]);
            }
        }
        // calculate epsilon and increase estimates for observation t
        for(size_t i = 0; i < numberOfStates; i++){
            for(size_t j = 0; j < numberOfStates; j++){
                logEpsilon[i][j] = sum_logarithms(logEpsilon[i][j],
                    logForward[i][t] +
                    m.mLogTransitions[i][j] +
                    m.mStates[j].logPdf(obs[t+1]) +
                    prevLogBackward[j]); // -logEvaluation (it simplifies)
                    //prevLogBackward is beta_{t+1}
            }
            // calculate gamma_t(i) for the current t
            // logBackward is beta_t
            logGamma[i][t] = logForward[i][t] + logBackward[i];
            logGammaSum[i] = sum_logarithms(logGammaSum[i], logGamma[i][t]);
            // update the accumulators
            logAverage[i] = sum_logarithms(logAverage[i],
                logGamma[i][t] + log(obs[t]-minObs));
        }
        // to avoid copying the array
        tmp = logBackward;
        logBackward = prevLogBackward;
        prevLogBackward = tmp;
    }
    // compute final reestimated parameters
    wahmm::real_t currentNewAverage;
    for(size_t i = 0; i < numberOfStates; i++){
        for(size_t j = 0; j < numberOfStates; j++){
            logEpsilon[i][j] -= logGammaSum[i]; // a_{ij}
        }
        logPi[i] = logGamma[i][0] - logEvaluation; // pi_i = gamma_1(i)
        logAverage[i] -= logGammaSum[i];
        currentNewAverage = exp(logAverage[i])+minObs;
        for(size_t t = 0; t < obs.size()-1; t++){
            logVariance[i] = sum_logarithms(logVariance[i],
                logGamma[i][t] + 2*log(abs(obs[t] - currentNewAverage)));
        }
        logVariance[i] -= logGammaSum[i];
    }

    // update parameters in the model
    for(size_t i = 0; i < numberOfStates; i++){
        m.mLogPi[i] = logPi[i];
        for(size_t j = 0; j < numberOfStates; j++){
            m.mLogTransitions[i][j] = logEpsilon[i][j];
        }
        m.mStates[i].updateParameters(exp(logAverage[i])+minObs,
            sqrt(exp(logVariance[i])));
    }

    freeMatrix(logForward, numberOfStates);

    return logEvaluation;
}

/**
* Wrapper for the training problem, to perform more iterations of the
* Baum-Welch algorithm.
*/
void training_problem_wrapper(Model& m, std::vector<wahmm::real_t>& obs, wahmm::real_t thresh,
    size_t maxIterations, bool verbose){

    wahmm::real_t **logEpsilon; // eps_t(i,j), accumulator over all t
    wahmm::real_t *logBackward = new wahmm::real_t[m.mStates.size()]; // only current t
    wahmm::real_t *prevLogBackward = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *logPi = new wahmm::real_t[m.mStates.size()]; // computed at last
    wahmm::real_t **logGamma;
    wahmm::real_t *logGammaSum = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *logAverage = new wahmm::real_t[m.mStates.size()]; // one for each state
    wahmm::real_t *logVariance = new wahmm::real_t[m.mStates.size()]; // one for each state

    logEpsilon = new wahmm::real_t*[m.mStates.size()];
    logGamma = new wahmm::real_t*[m.mStates.size()];
    for(size_t i = 0; i < m.mStates.size(); i++){
        logEpsilon[i] = new wahmm::real_t[m.mStates.size()];
        logGamma[i] = new wahmm::real_t[obs.size()-1];
    }


    wahmm::real_t minObs = 0;
    for(auto it = obs.begin(); it != obs.end(); it++){
        if(*it < minObs)
            minObs = *it;
    }
    minObs -= 1; // to avoid crash when 0 is saves as -0.0000000001
    wahmm::real_t evaluation=-infin, newEvaluation=-infin;
    wahmm::real_t logImprovement = thresh + 1;
    size_t iter;
    if(verbose)
        std::cout << "+++++ Training problem +++++" << std::endl;
    for(iter = 0; iter < maxIterations && logImprovement > thresh; iter++){
        newEvaluation = training_problem(m, obs, minObs,
            logEpsilon, logBackward, prevLogBackward, logPi, logGamma,
            logGammaSum, logAverage, logVariance, verbose);
        // newEvaluation = training_problem_scaled(m, obs, minObs,
        //     logEpsilon, logPi, logGamma,
        //     logGammaSum, logAverage, logVariance);
        logImprovement = newEvaluation - evaluation;
        evaluation = newEvaluation;
    }
    if(verbose){
        std::cout << "Number of iterations: " << iter << std::endl;
        m.printModel();
    }



    // free variables
    freeMatrix(logEpsilon, m.mStates.size());
    freeMatrix(logGamma, m.mStates.size());
    delete[] logBackward;
    delete[] prevLogBackward;
    delete[] logPi;
    delete[] logGammaSum;
    delete[] logAverage;
    delete[] logVariance;
}

#endif