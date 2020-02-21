#ifndef WAHMM_ALGORITHMS_COMPRESSED_HPP
#define WAHMM_ALGORITHMS_COMPRESSED_HPP

#include "Compressor.hpp"
#include "commons.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

void evaluation_compressed(Model& m, Compressor *c, bool verbose);
wahmm::real_t** forward_matrix_compressed(Model& m, Compressor *c);

/**
* Solve the evaluation problem through the forward algorithm.
* If verbose is true, print the results.
*
* @param m the model
* @param obs the observation sequence
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
* @param obs the observation sequence
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

#endif
