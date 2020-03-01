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
    // std::cout << "Starting termination..." << std::endl;
    // termination
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][c->blocksNumber()-1]);
    }
    // std::cout << "Termination done." << std::endl;

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

    // std::cout << "Initiating FMC..." << std::endl;
    c->initForward();
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        // two elements (start, end) per block
        logForward[i] = new wahmm::real_t[c->blocksNumber()];
        // alpha_0(i) = pi_i * E_1(i)
        logForward[i][0] = m.mLogPi[i] + compute_e(m, i, c->blockData());
    }
    // std::cout << "Initialization done. Starting induction..." << std::endl;
    // induction
    size_t blockCounter = 1;
    while(c->next()){
        // std::cout << "Iteration with blockCounter=" << blockCounter << std::endl;
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
    // std::cout << "Induction done." << std::endl;
    c->initForward();

    // std::cout << "Exiting FMC..." << std::endl;
    return logForward;
}


wahmm::real_t** backward_matrix_compressed(Model& m, Compressor *c){
    wahmm::real_t **logBackward;
    size_t numberOfStates = m.mStates.size();
    logBackward = new wahmm::real_t*[numberOfStates]; // forward variables


    c->initBackward();
    int blockCounter = c->blocksNumber() - 1;
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logBackward[i] = new wahmm::real_t[c->blocksNumber()];
        logBackward[i][blockCounter] = 1;
    }
    blockCounter--;
    // induction
    while(blockCounter >= 0){
        for(size_t i = 0; i < numberOfStates; i++){ // arriving state
            logBackward[i][blockCounter] = -infin;
            for(size_t j = 0; j < numberOfStates; j++){ // starting state
                // beta_t(i) = sum_{j=1}^N a_{ij} b_j(O_{t+1}) beta_{t+1}(j)
                logBackward[i][blockCounter] = sum_logarithms(logBackward[i][blockCounter],
                    m.mLogTransitions[i][j] +
                    compute_e(m, j, c->reverseBlockData()) +
                    logBackward[j][blockCounter+1]);
            }
        }
        c->reverseNext();
        blockCounter--;
    }
    c->initBackward();

    return logBackward;
}



void decoding_compressed(Model &m, Compressor *c, bool verbose){
    wahmm::real_t **logViterbi, **statesViterbi;
    size_t numberOfStates = m.mStates.size();
    logViterbi = new wahmm::real_t*[numberOfStates];
    statesViterbi = new wahmm::real_t*[numberOfStates];
    std::vector<size_t> blockLengths;
    if(verbose)
        std::cout << "+++++ Decoding Compressed Problem +++++" << std::endl;

    c->initForward();
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logViterbi[i] = new wahmm::real_t[c->blocksNumber()];
        statesViterbi[i] = new wahmm::real_t[c->blocksNumber()];
        // delta_0(i) = pi_i * b_i(O_1)
        logViterbi[i][0] = m.mLogPi[i] + compute_e(m, i, c->blockData());
        statesViterbi[i][0] = -1; // psi_0(i) = 0
    }
    blockLengths.push_back(c->blockSize());
    // induction
    wahmm::real_t currentMax = -infin;
    wahmm::real_t currentSum = 0;
    int blockCounter = 1;
    int currentState = -1;
    while(c->next()){ // observations
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
            logViterbi[j][blockCounter] = currentMax + compute_e(m, j, c->blockData());
            // psi_t(j) = argmax[...]
            statesViterbi[j][blockCounter] = currentState;
            // re-initialize max variables for next loop
            currentMax = -infin;
            currentState = -1;
        }
        blockLengths.push_back(c->blockSize());
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
    for(; blockCounter > 0; blockCounter--){
        // if currentState == -1, impossible path? can it happen?
        if(currentState >= 0)
            currentState = statesViterbi[currentState][blockCounter];
        else
            std::cerr << "[Error] Impossible Viterbi path!" << std::endl;
        viterbiPath.push_front(currentState);
    }

    // print results
    if(verbose){
        printMatrixSummary(logViterbi, numberOfStates, c->blocksNumber(),
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
    std::ofstream ofs ("results/wahmm_viterbi_compressed", std::ofstream::out);
    size_t lenIndex = 0;
    for(auto it = viterbiPath.begin(); it != viterbiPath.end(); it++){
        for(size_t blen = 0; blen < blockLengths[lenIndex]; blen++){
            ofs << *it << " ";
        }
        lenIndex++;
    }
    ofs.close();

    c->initForward();

    freeMatrix(logViterbi, numberOfStates);
    freeMatrix(statesViterbi, numberOfStates);
}




wahmm::real_t training_compressed(Model& m, Compressor *c, wahmm::real_t minSum,
    wahmm::real_t **logEpsilon,
    wahmm::real_t *logPi, wahmm::real_t **logGamma, wahmm::real_t *logGammaSum,
    wahmm::real_t *logTrDen,
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
        logGammaSum[i] = -infin;
        logAverage[i] = -infin;
        logVariance[i] = -infin;
    }

    logForward = forward_matrix_compressed(m, c);
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][c->blocksNumber()-1]);
    }
    logBackward = backward_matrix_compressed(m, c);

    c->initForward();
    // iterate over blocks
    int bc = 0; // block counter
    size_t nBlocks = c->blocksNumber();
    blockdata currentBd = c->blockData();

    wahmm::real_t logEvalGammaSum = -infin;
    for(int i = 0; i < numberOfStates; i++){
        logEvalGammaSum = sum_logarithms(logEvalGammaSum,
            logForward[i][0] +
            logBackward[i][0]);
    }

    while(c->next()){ // "current block" is w+1
        for(int i = 0; i < numberOfStates; i++){
            for(int j = 0; j < numberOfStates; j++){
                logEpsilon[i][j] = sum_logarithms(logEpsilon[i][j],
                    logForward[i][bc] +
                    m.mLogTransitions[i][j] +
                    compute_e(m, j, c->blockData()) +
                    logBackward[j][bc+1]);
                if(i == j){
                    logEpsilon[i][j] = sum_logarithms(logEpsilon[i][j],
                        log(currentBd.nw - 1) +
                        logForward[i][bc] +
                        logBackward[i][bc]);
                }
            }
            logGamma[i][bc] = logForward[i][bc] + logBackward[i][bc];
            logGammaSum[i] = sum_logarithms(logGammaSum[i],
                log(currentBd.nw) + logGamma[i][bc]);
        }
        currentBd = c->blockData();
        bc++;
    }
    // last block was not processed
    for(int i = 0; i < numberOfStates; i++){
        logEpsilon[i][i] = sum_logarithms(logEpsilon[i][i],
            log(currentBd.nw - 1) +
            logForward[i][bc] +
            logBackward[i][bc]);
        logGamma[i][bc] = logForward[i][bc] + logBackward[i][bc];
        logGammaSum[i] = sum_logarithms(logGammaSum[i],
            log(currentBd.nw) + logGamma[i][bc]);
    }

    // reestimated parameters
    for(int i = 0; i < numberOfStates; i++){
        logPi[i] = logGamma[i][0] - logEvalGammaSum;
        for(int j = 0; j < numberOfStates; j++){
            logEpsilon[i][j] -= logGammaSum[i];
        }
        c->initForward();
        logAverage[i] = 0;
        wahmm::real_t v;
        for(int b = 0; b < nBlocks; b++){
            currentBd = c->blockData();
            v = currentBd.s1;
            logAverage[i] += exp(logGamma[i][b]-logGammaSum[i]) * v;
            c->next();
        }
        c->initForward();
        logVariance[i] = 0;
        for(int b = 0; b < nBlocks; b++){
            currentBd = c->blockData();
            v = currentBd.s2 - 2*logAverage[i]*currentBd.s1 + currentBd.nw*logAverage[i]*logAverage[i];
            logVariance[i] += exp(logGamma[i][b]-logGammaSum[i]) * v;
            c->next();
        }
    }

    //update model
    for(size_t i = 0; i < numberOfStates; i++){
        m.mLogPi[i] = logPi[i];
        for(size_t j = 0; j < numberOfStates; j++){
            m.mLogTransitions[i][j] = logEpsilon[i][j];
        }
        m.mStates[i].updateParameters(logAverage[i],
            sqrt(logVariance[i]));
    }

    freeMatrix(logForward, numberOfStates);
    freeMatrix(logBackward, numberOfStates);

    return logEvaluation;
}



void training_compressed_wrapper(Model& m, Compressor *c, wahmm::real_t thresh,
    size_t maxIterations, bool verbose){

    wahmm::real_t **logEpsilon; // eps_t(i,j), accumulator over all t
    wahmm::real_t *logBackward = new wahmm::real_t[m.mStates.size()]; // only current t
    wahmm::real_t *prevLogBackward = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *logPi = new wahmm::real_t[m.mStates.size()]; // computed at last
    wahmm::real_t **logGamma;
    wahmm::real_t *logTrDen = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *logGammaSum = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *logAverage = new wahmm::real_t[m.mStates.size()]; // one for each state
    wahmm::real_t *logVariance = new wahmm::real_t[m.mStates.size()]; // one for each state

    logEpsilon = new wahmm::real_t*[m.mStates.size()];
    logGamma = new wahmm::real_t*[m.mStates.size()];
    for(size_t i = 0; i < m.mStates.size(); i++){
        logEpsilon[i] = new wahmm::real_t[m.mStates.size()];
        logGamma[i] = new wahmm::real_t[c->blocksNumber()];
    }

    c->buildReverse();
    c->initForward();
    wahmm::real_t minSum = 0;
    do {
        if(c->blockData().s1 < minSum)
            minSum = c->blockData().s1;
    } while(c->next());
    c->initForward();
    // std::cout << "minSum: " << minSum << std::endl;
    minSum -= 1; // to avoid crash when 0 is saves as -0.0000000001
    wahmm::real_t evaluation=-infin, newEvaluation=-infin;
    wahmm::real_t logImprovement = thresh + 1;
    size_t iter;
    if(verbose)
        std::cout << "+++++ Training Compressed Problem +++++" << std::endl;
    for(iter = 0; iter < maxIterations && logImprovement > thresh; iter++){
        newEvaluation = training_compressed(m, c, minSum,
            logEpsilon, logPi, logGamma,
            logGammaSum, logTrDen, logAverage, logVariance, verbose);
        logImprovement = newEvaluation - evaluation;
        evaluation = newEvaluation;
        m.printModel();
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
