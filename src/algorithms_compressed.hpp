#ifndef WAHMM_ALGORITHMS_COMPRESSED_HPP
#define WAHMM_ALGORITHMS_COMPRESSED_HPP

#include "Compressor.hpp"
#include "commons.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

void evaluation_compressed(Model& m, Compressor *c, bool verbose);
wahmm::real_t** forward_matrix_compressed(Model& m, Compressor *c);
// void decoding_compressed(Model &m, Compressor *c, bool verbose);

/**
* Solve the evaluation problem through the forward algorithm.
* If verbose is true, print the results.
*
* @param m the model
* @param c compressor holding the HaMMLET interface
* @param verbose if true prints result informations
*/
void evaluation_compressed(Model& m, Compressor *c, bool verbose){
    wahmm::real_t **logForward;
    wahmm::real_t logEvaluation;
    size_t numberOfStates = m.mStates.size();

    if(verbose)
        std::cout << "+++++ Evaluation Compressed Problem +++++" << std::endl;

    logForward = forward_matrix_compressed(m, c); // initialization and induction
    // termination
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][c->blocksNumber()-1]);
    }

    // print results
    if(verbose){
        printMatrixSummary(logForward, numberOfStates, c->blocksNumber(),
            "logForward compressed", false);
        std::cout << "log[ P(O|lambda) ] = " << logEvaluation << std::endl;
    }

    freeMatrix(logForward, numberOfStates);
}


wahmm::real_t** forward_matrix_compressed(Model& m, Compressor *c){
    wahmm::real_t **logForward;
    size_t numberOfStates = m.mStates.size();
    logForward = new wahmm::real_t*[numberOfStates]; // forward variables

    c->initForward();
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        // two elements (start, end) per block
        logForward[i] = new wahmm::real_t[c->blocksNumber()];
        // alpha_0(i) = pi_i * E_1(i)
        logForward[i][0] = m.mLogPi[i] + compute_e(m, i, c->blockData());
    }
    // induction
    size_t blockCounter = 1;
    while(c->next()){
        for(size_t j = 0; j < numberOfStates; j++){ // arriving state
            logForward[j][blockCounter] = -infin;
            for(int i = 0; i < numberOfStates; i++){ // starting state
                // alpha_W(j) = sum_{i=0}^N alpha_{w-1}(i)a_{ij} ...
                logForward[j][blockCounter] = sum_logarithms(logForward[j][blockCounter],
                    logForward[i][blockCounter-1] + m.mLogTransitions[i][j]);
            }
            // ... E_w(j)
            logForward[j][blockCounter] += compute_e(m, j, c->blockData());
        }
        blockCounter++;
    }

    c->initForward();

    return logForward;
}

/*
wahmm::real_t** backward_matrix_compressed(Model& m, Compressor *c){
    wahmm::real_t **logBackward;
    size_t numberOfStates = m.mStates.size();
    logBackward = new wahmm::real_t*[numberOfStates]; // forward variables

    c->initBackward();
    int blockCounter = c->blocksNumber() - 1;
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logBackward[i] = new wahmm::real_t[c->blocksNumber()];
        logBackward[i][blockCounter--] = compute_e(m, i, c->reverseBlockData());
    }
    // induction
    while(c->reverseNext() && blockCounter >= 0){ //c->reverseNext() will never return false because the last block should not be considered
        for(size_t i = 0; i < numberOfStates; i++){ // arriving state
            logBackward[i][blockCounter] = -infin;
            for(size_t j = 0; j < numberOfStates; j++){ // starting state
                // beta_t(i) = sum_{j=1}^N a_{ij} b_j(O_{t+1}) beta_{t+1}(j)
                logBackward[i][blockCounter] = sum_logarithms(logBackward[i][blockCounter],
                    m.mLogTransitions[i][j] +
                    m.mStates[j].logPdf(c->reverseAvg()) +
                    logBackward[j][blockCounter+1]);
            }
        }
        blockCounter--;
        for(size_t i = 0; i < numberOfStates; i++){
            logBackward[i][blockCounter] = logBackward[i][blockCounter+1] +
                (c->reverseSize() - 1) * (m.mLogTransitions[i][i] + m.mStates[i].logPdf(c->reverseAvg()));
        }
        blockCounter--;
    }

    c->initBackward();

    return logBackward;
}

/*

void decoding_compressed(Model &m, Compressor *c, bool verbose){
    wahmm::real_t **logViterbi, **statesViterbi;
    size_t numberOfStates = m.mStates.size();
    logViterbi = new wahmm::real_t*[numberOfStates];
    statesViterbi = new wahmm::real_t*[numberOfStates];
    std::vector<size_t> stateDurations;

    if(verbose)
        std::cout << "+++++ Decoding Compressed Problem +++++" << std::endl;

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logViterbi[i] = new wahmm::real_t[2*c->blocksNumber()];
        statesViterbi[i] = new wahmm::real_t[c->blocksNumber()];
        // delta_0(i) = pi_i * b_i(O_1)
        logViterbi[i][0] = m.mLogPi[i] + m.mStates[i].logPdf(c->blockAvg());
        statesViterbi[i][0] = -1; // psi_0(i) = 0
        stateDurations.push_back(c->blockSize());
        logViterbi[i][1] = logViterbi[i][0] + (c->blockSize()-1) *
            (m.mLogTransitions[i][i] + m.mStates[i].logPdf(c->blockAvg()));
    }
    // induction
    wahmm::real_t currentMax = -infin;
    wahmm::real_t currentSum = 0;
    size_t blockCounter = 2;
    size_t currentState = -1;
    while(c->next()){ // observations
        // std::cout << "blockCounter: " << blockCounter << std::endl;
        for(size_t j = 0; j < numberOfStates; j++){ // arriving state
            logViterbi[j][blockCounter] = -infin;
            // delta_t(j) = max_{1<=i<=N} delta_{t-1}(i)a_{ij} ...
            for(size_t i = 0; i < numberOfStates; i++){ // starting state
                currentSum = logViterbi[i][blockCounter-1] + m.mLogTransitions[i][j];
                if(currentSum > currentMax){
                    currentMax = currentSum;
                    currentState = i;
                }
            }
            // ... b_{j}(O_{t})
            logViterbi[j][blockCounter] = currentMax + m.mStates[j].logPdf(c->blockAvg());
            // psi_t(j) = argmax[...]
            statesViterbi[j][blockCounter/2] = currentState;
            // re-initialize max variables for next loop
            currentMax = -infin;
            currentState = -1;
        }
        blockCounter++;
        for(size_t j = 0; j < numberOfStates; j++){
            logViterbi[j][blockCounter] = logViterbi[j][blockCounter-1] + (c->blockSize()-1) *
                (m.mLogTransitions[j][j] + m.mStates[j].logPdf(c->blockAvg()));
        }
        blockCounter++;
    }
    // termination
    wahmm::real_t logDecoding;
    list<size_t> viterbiPath;
    currentMax = -infin; // this will contain the log probability of the path
    currentState = -1;
    blockCounter--;
    for(size_t i = 0; i < numberOfStates; i++){
        if(logViterbi[i][blockCounter] > currentMax){
            currentMax = logViterbi[i][blockCounter];
            currentState = i;
        }
    }
    viterbiPath.push_front(currentState);
    for(blockCounter--; blockCounter > 0; blockCounter -= 2){
        // if currentState == -1, impossible path? can it happen?
        if(currentState >= 0)
            currentState = statesViterbi[currentState][blockCounter/2];
        else
            std::cerr << "[Error] Impossible Viterbi path!" << std::endl;
        viterbiPath.push_front(currentState);
    }

    // print results
    if(verbose){
        printMatrixSummary(logViterbi, numberOfStates, 2*c->blocksNumber(),
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

    c->initForward();

    freeMatrix(logViterbi, numberOfStates);
    freeMatrix(statesViterbi, numberOfStates);
}






wahmm::real_t training_compressed(Model& m, Compressor *c, wahmm::real_t minObs,
    wahmm::real_t **logEpsilon,
    wahmm::real_t *logPi, wahmm::real_t **logGamma, wahmm::real_t *logGammaSum,
    wahmm::real_t *logAverage, wahmm::real_t *logVariance, bool verbose){

    wahmm::real_t logEvaluation; // P(O|lambda)
    wahmm::real_t **logForward; // forward matrix
    wahmm::real_t **logBackward; // backward matrix
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

    logForward = forward_matrix_compressed(m, c);
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][2*c->blocksNumber() - 1]);
    }
    logBackward = backward_matrix_compressed(m, c);

    // only consider the end of the blocks as representative for the block
    for(size_t bc = 1; bc < 2*c->blocksNumber() - 1; bc += 2){
        for(size_t i = 0; i < numberOfStates; i++){
            logGamma[i][bc/2] = logForward[i][bc] + logBackward[i][bc];
            logGammaSum[i] = sum_logarithms(logGammaSum[i], logGamma[i][bc/2]);
            // update the accumulators
            logAverage[i] = sum_logarithms(logAverage[i],
                logGamma[i][bc/2] + log(c->blockSum()-minSum));
        }
        // intra block
        for(size_t i = 0; i < numberOfStates; i++){
            logEpsilon[i][i] = sum_logarithms(logEpsilon[i][i],
                logForward[i][bc] +
                m.mLogTransitions[i][i] +
                m.mStates[i].logPdf(obs[bc+1]) +
                logBackward[i][bc+1]); // -logEvaluation (it simplifies)
                //prevLogBackward is beta_{t+1}
        }
        // inter blocks
        for(size_t i = 0; i < numberOfStates; i++){
            for(size_t j = 0; j < numberOfStates; j++){
                logEpsilon[i][j] = sum_logarithms(logEpsilon[i][j],
                    m.mStates[j].logPdf(obs[t+1]) +
                    logBackward[j][bc+1]); // -logEvaluation (it simplifies)
                    //prevLogBackward is beta_{t+1}
                    if(i == j){ //also add intrablock summary
                        logEpsilon[i][j] = sum_logarithms(logEpsilon[i][j],
                            logForward[i][bc] +
                            m.mLogTransitions[i][j] +
                            m.mStates[j].logPdf(obs[t+1]) +
                            logBackward[j][bc+1]);
                    }
            }
        }
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



void training_compressed_wrapper(Model& m, std::vector<wahmm::real_t>& obs, wahmm::real_t thresh,
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
*/

#endif
