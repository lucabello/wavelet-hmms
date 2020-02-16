#ifndef WAHMM_ALGORITHMS_HPP
#define WAHMM_ALGORITHMS_HPP
#include "includes.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

void evaluation_problem(Model& m, vector<real_t>& obs, bool verbose);
real_t** forward_matrix(Model& m, vector<real_t>& obs);
void decoding_problem(Model &m, vector<real_t>& obs, bool verbose);

/**
* Solve the evaluation problem through the forward algorithm.
* If verbose is true, print the results.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true prints result informations
*/
void evaluation_problem(Model& m, vector<real_t>& obs, bool verbose){
    real_t **logForward;
    real_t logEvaluation;

    if(verbose)
        cout << "+++++ Evaluation Problem +++++" << endl;

    logForward = forward_matrix(m, obs); // initialization and induction
    // termination
    logEvaluation = -inf;
    for(int i = 0; i < m.mStates.size(); i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][obs.size()-1]);
    }

    // print results
    if(verbose){
        printMatrixSummary(logForward, m.mStates.size(), obs.size(),
            "logForward", false);
        cout << "log[ P(O|lambda) ] = " << logEvaluation << endl;
    }

    freeMatrix(logForward, m.mStates.size());
}

/**
* Compute the forward matrix given a model and an observations sequence.
*
* @param m the model
* @param obs the observation sequence
*/
real_t** forward_matrix(Model& m, vector<real_t>& obs){
    real_t **logForward;
    logForward = new real_t*[m.mStates.size()]; // forward variables

    // initialization
    for(int i = 0; i < m.mStates.size(); i++){
        logForward[i] = new real_t[obs.size()];
        // alpha_0(i) = pi_i * b_i(O_1)
        logForward[i][0] = m.mInitialDistribution[i] + m.mStates[i]
            .distribution().pdf(obs[0]);
    }

    // induction
    for(int o = 1; o < obs.size(); o++){ // observations
        for(int j = 0; j < m.mStates.size(); j++)
            logForward[j][o] = -inf;
        for(int j = 0; j < m.mStates.size(); j++){ // arriving state
            for(int i = 0; i < m.mStates.size(); i++){ // starting state
                // alpha_t+1(j) = sum_{i=0}^N alpha_t(i)a_{ij} ...
                logForward[j][o] = sum_logarithms(logForward[j][o],
                    logForward[i][o-1] + m.mLogTransitions[i][j]);
            }
            // ... b_{j}(O_{t+1})
            logForward[j][o] += m.mStates[j].distribution().pdf(obs[o]);
        }
    }

    return logForward;
}

/**
* Compute the forward backward given a model and an observations sequence.
*
* @param m the model
* @param obs the observation sequence
*/
real_t** backward_matrix(Model& m, vector<real_t>& obs){
    real_t **logBackward;
    logBackward = new real_t*[m.mStates.size()]; // forward variables

    // initialization
    for(int i = 0; i < m.mStates.size(); i++){
        logBackward[i] = new real_t[obs.size()];
        // beta_T(i) = 1
        logBackward[i][obs.size()-1] = 0;
    }

    // induction
    for(int t = obs.size()-2; t >= 0; t--){ // observations
        for(int j = 0; j < m.mStates.size(); j++)
            logBackward[j][t] = -inf;
        for(int i = 0; i < m.mStates.size(); i++){ // arriving state
            for(int j = 0; j < m.mStates.size(); j++){ // starting state
                // beta_t(i) = sum_{j=1}^N a_{ij} b_j(O_{t+1}) beta_{t+1}(j)
                logBackward[i][t] = sum_logarithms(logBackward[i][t],
                    m.mLogTransitions[i][j] +
                    m.mStates[j].distribution().pdf(obs[t+1]) +
                    logBackward[j][t+1]);
            }
        }
    }

    return logBackward;
}


/**
* Solve the evaluation problem through the forward algorithm.
* Currently write the obtained path in results/wahmm_viterbi.
* If verbose is true print the Viterbi matrix.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true print result informations
*/
void decoding_problem(Model &m, vector<real_t>& obs, bool verbose){
    real_t **logViterbi, **statesViterbi;
    logViterbi = new real_t*[m.mStates.size()];
    statesViterbi = new real_t*[m.mStates.size()];

    if(verbose)
        cout << "+++++ Decoding Problem +++++" << endl;

    // initialization
    for(int i = 0; i < m.mStates.size(); i++){
        logViterbi[i] = new real_t[obs.size()];
        statesViterbi[i] = new real_t[obs.size()];
        // delta_0(i) = pi_i * b_i(O_1)
        logViterbi[i][0] = m.mInitialDistribution[i] + m.mStates[i]
            .distribution().pdf(obs[0]);
        // psi_0(i) = 0
        statesViterbi[i][0] = -1;
    }

    // induction
    real_t currentMax = -inf;
    real_t currentSum = 0;
    size_t currentState = -1;
    for(int o = 1; o < obs.size(); o++){ // observations
        for(int j = 0; j < m.mStates.size(); j++)
            logViterbi[j][o] = -inf;
        for(int j = 0; j < m.mStates.size(); j++){ // arriving state
            // delta_t(j) = max_{1<=i<=N} delta_{t-1}(i)a_{ij} ...
            for(int i = 0; i < m.mStates.size(); i++){ // starting state
                currentSum = logViterbi[i][o-1] + m.mLogTransitions[i][j];
                if(currentSum > currentMax){
                    currentMax = currentSum;
                    currentState = i;
                }
            }
            // ... b_{j}(O_{t})
            logViterbi[j][o] = currentMax + m.mStates[j]
                .distribution().pdf(obs[o]);
            // psi_t(j) = argmax[...]
            statesViterbi[j][o] = currentState;
            // re-initialize max variables for next loop
            currentMax = -inf;
            currentState = -1;
        }
    }

    // termination
    real_t logDecoding;
    list<size_t> viterbiPath;
    currentMax = -inf; // this will contain the log probability of the path
    currentState = -1;
    for(int i = 0; i < m.mStates.size(); i++){
        if(logViterbi[i][obs.size()-1] > currentMax){
            currentMax = logViterbi[i][obs.size()-1];
            currentState = i;
        }
    }

    viterbiPath.push_front(currentState);
    for(int o = obs.size()-2; o >= 0; o--){
        // if currentState == -1, impossible path? can it happen?
        if(currentState >= 0)
            currentState = statesViterbi[currentState][o + 1];
        viterbiPath.push_front(currentState);
    }

    // print results
    if(verbose){
        printMatrixSummary(logViterbi, m.mStates.size(), obs.size(),
            "logViterbi", false);
        cout << "Most likely path: " << endl;
        int i = 0;
        for(auto it = viterbiPath.begin(); it != viterbiPath.end(); it++, i++){
            // only print first 5 and last 5 states
            if(i == 5)
                cout << "... ";
            if(i >= 5 && i < viterbiPath.size() - 5)
                continue;
            cout << *it << " ";
        }
        cout << endl;
        cout << "log[ P(Q|O,lambda) ] = " << currentMax << endl;
    }

    // print to file for comparison with other implementations
    std::ofstream ofs ("results/wahmm_viterbi", std::ofstream::out);
    for(auto it = viterbiPath.begin(); it != viterbiPath.end(); it++)
        ofs << *it << " ";
    ofs.close();


    for(int i = 0; i < m.mStates.size(); i++){
        delete[] logViterbi[i];
        delete[] statesViterbi[i];
    }
    delete[] logViterbi;
    delete[] statesViterbi;
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
real_t training_problem(Model& m, vector<real_t>& obs, real_t minObs,
    real_t **logEpsilon, real_t *logBackward,
    real_t *prevLogBackward, real_t *logPi, real_t *logGamma,
    real_t *currentGamma, real_t *logAverage, real_t *logVariance){

    real_t logEvaluation; // P(O|lambda)
    real_t **logForward; // forward matrix
    size_t numberOfStates = m.mStates.size();

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        for(size_t j = 0; j < numberOfStates; j++)
            logEpsilon[i][j] = -inf;
        logBackward[i] = 0;
        prevLogBackward[i] = 0;
        logGamma[i] = -inf;
        logAverage[i] = -inf;
        logVariance[i] = -inf;
    }
    logForward = forward_matrix(m, obs);
    logEvaluation = -inf;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][obs.size()-1]);
    }

    // start from T-1
    for(int t = obs.size()-2; t >= 0; t--){
        //cout << "[Debug] Starting observation " << t << endl;
        // calculate backward variable for the next iteration
        real_t currentSum;
        for(int i = 0; i < numberOfStates; i++){
            logBackward[i] = -inf;
            for(size_t j = 0; j < numberOfStates; j++){
                currentSum = m.mLogTransitions[i][j] +
                    m.mStates[j].distribution().pdf(obs[t+1]) +
                    prevLogBackward[j];
                logBackward[i] = sum_logarithms(logBackward[i], currentSum);
            }

        }
        //cout << "[Debug] Backward("<<t<<") = "<< logBackward[0] << "," << logBackward[1] << endl;
        //cout << "[Debug] Backward variable computed " << endl;
        // calculate epsilon and increase estimates for observation t
        for(size_t i = 0; i < numberOfStates; i++){
            for(size_t j = 0; j < m.mStates.size(); j++){
                logEpsilon[i][j] = sum_logarithms(logEpsilon[i][j],
                    logForward[i][t] + m.mLogTransitions[i][j] +
                    m.mStates[j].distribution().pdf(obs[t+1]) +
                    prevLogBackward[j]); // -logEvaluation (it simplifies ?)
                    //prevLogBackward is beta_{t+1}
            }
            // calculate gamma_t(i) for the current t
            // logBackward is beta_t
            currentGamma[i] = logForward[i][t] + logBackward[i];
            // update the accumulators
            logGamma[i] = sum_logarithms(logGamma[i], currentGamma[i]);
            logAverage[i] = sum_logarithms(logAverage[i],
                currentGamma[i] + log(obs[t]-minObs));
            // logVariance[i] = sum_logarithms(logVariance[i],
            //     currentGamma[i] + log( pow(obs[t] - m.mStates[i]
            //     .distribution().mean(),2) ));
            logVariance[i] = sum_logarithms(logVariance[i],
                currentGamma[i] + 2*log( abs(obs[t] - m.mStates[i]
                .distribution().mean()) ));
        }
        real_t *tmp;
        // to avoid copying the array
        tmp = logBackward;
        logBackward = prevLogBackward;
        prevLogBackward = tmp;
    }

    // compute final reestimated parameters
    for(size_t i = 0; i < numberOfStates; i++){
        logPi[i] = currentGamma[i] - logEvaluation; // pi_i = gamma_1(i)
        for(size_t j = 0; j < m.mStates.size(); j++){
            logEpsilon[i][j] -= logGamma[i]; // a_{ij}
        }
        logAverage[i] -= logGamma[i]; // mu_j
        logVariance[i] -= logGamma[i]; // sigma^2_j
    }

    // update parameters in the model
    for(size_t i = 0; i < numberOfStates; i++){
        m.mInitialDistribution[i] = logPi[i];
        for(size_t j = 0; j < numberOfStates; j++){
            m.mLogTransitions[i][j] = logEpsilon[i][j];
        }
        m.mStates[i].setDistribution(NormalDistribution(
            exp(logAverage[i])+minObs, sqrt(exp(logVariance[i]))));
    }

    cout << "[Debug] Variances: "<<exp(logVariance[0])<<","<<exp(logVariance[1])<<endl;
    freeMatrix(logForward, numberOfStates);

    return logEvaluation;
}


void training_problem_wrapper(Model& m, vector<real_t>& obs, real_t thresh,
    size_t maxIterations){

    real_t **logEpsilon; // eps_t(i,j), accumulator over all t
    real_t *logBackward = new real_t[m.mStates.size()]; // only current t
    real_t *prevLogBackward = new real_t[m.mStates.size()];
    real_t *logPi = new real_t[m.mStates.size()]; // computed at last
    real_t *logGamma = new real_t[m.mStates.size()]; // computed at last
    real_t *currentGamma = new real_t[m.mStates.size()]; // only current t
    real_t *logAverage = new real_t[m.mStates.size()]; // one for each state
    real_t *logVariance = new real_t[m.mStates.size()]; // one for each state

    logEpsilon = new real_t*[m.mStates.size()];
    for(size_t i = 0; i < m.mStates.size(); i++){
        logEpsilon[i] = new real_t[m.mStates.size()];
    }


    real_t minObs = 0;
    for(auto it = obs.begin(); it != obs.end(); it++){
        if(*it < minObs)
            minObs = *it;
    }
    minObs -= 1; // to avoid crash when 0 is saves as -0.0000000001
    real_t evaluation=-inf, newEvaluation=-inf;
    real_t logImprovement = thresh + 1;
    size_t iter;
    for(iter = 0; iter < maxIterations && logImprovement > thresh; iter++){
        newEvaluation = training_problem(m, obs, minObs,
            logEpsilon, logBackward, prevLogBackward, logPi, logGamma,
            currentGamma, logAverage, logVariance);
        cout << "newEvaluation: " << newEvaluation << endl;
        logImprovement = newEvaluation - evaluation;
        evaluation = newEvaluation;
        m.printModel();
    }
    cout << "Number of iterations: " << iter << endl;


    // free variables
    freeMatrix(logEpsilon, m.mStates.size());
    delete[] logPi;
    delete[] logGamma;
    delete[] logAverage;
    delete[] logVariance;
}

#endif
