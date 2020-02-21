#ifndef WAHMM_ALGORITHMS_COMPRESSED_HPP
#define WAHMM_ALGORITHMS_COMPRESSED_HPP

#include "Compressor.hpp"
#include "commons.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

void evaluation_compressed(Model& m, Compressor *c, bool verbose);
wahmm::real_t** forward_matrix_compressed(Model& m, Compressor *c);
void decoding_compressed(Model &m, Compressor *c, bool verbose);

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
            logForward[i][2*c->blocksNumber()-1]);
    }

    // print results
    if(verbose){
        printMatrixSummary(logForward, numberOfStates, c->blocksNumber(),
            "logForward compressed", false);
        std::cout << "log[ P(O|lambda) ] = " << logEvaluation << std::endl;
    }

    c->initForward();

    freeMatrix(logForward, numberOfStates);
}

/**
* Compute the forward matrix given a model and an observations sequence.
*
* @param m the model
* @param c compressor holding the HaMMLET interface
*/
wahmm::real_t** forward_matrix_compressed(Model& m, Compressor *c){
    wahmm::real_t **logForward;
    size_t numberOfStates = m.mStates.size();
    logForward = new wahmm::real_t*[numberOfStates]; // forward variables

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        // two elements (start, end) per block
        logForward[i] = new wahmm::real_t[2 * c->blocksNumber()];
        // alpha_0(i) = pi_i * b_i(O_1)
        logForward[i][0] = m.mLogPi[i] +
            m.mStates[i].logPdf(c->blockAvg());
        logForward[i][1] = logForward[i][0] + (c->blockSize()-1) *
            (m.mLogTransitions[i][i] + m.mStates[i].logPdf(c->blockAvg()));
    }

    // induction
    size_t blockCounter = 2;
    while(c->next()){
        for(size_t j = 0; j < numberOfStates; j++){ // arriving state
            logForward[j][blockCounter] = -infin;
            for(int i = 0; i < numberOfStates; i++){ // starting state
                // alpha_t+1(j) = sum_{i=0}^N alpha_t(i)a_{ij} ...
                logForward[j][blockCounter] = sum_logarithms(logForward[j][blockCounter],
                    logForward[i][blockCounter-1] + m.mLogTransitions[i][j]);
            }
            // ... b_{j}(O_{t+1})
            logForward[j][blockCounter] += m.mStates[j].logPdf(c->blockAvg());
        }
        blockCounter++;
        for(size_t j = 0; j < numberOfStates; j++){
            logForward[j][blockCounter] = logForward[j][blockCounter-1] + (c->blockSize()-1) *
                (m.mLogTransitions[j][j] + m.mStates[j].logPdf(c->blockAvg()));
        }
        blockCounter++;
    }

    return logForward;
}


/**
* Solve the decoding problem through the forward algorithm.
* Currently write the obtained path in results/wahmm_viterbi.
* If verbose is true print the Viterbi matrix.
*
* @param m the model
* @param c compressor holding the HaMMLET interface
* @param verbose if true print result informations
*/
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

    freeMatrix(logViterbi, numberOfStates);
    freeMatrix(statesViterbi, numberOfStates);
}



#endif
