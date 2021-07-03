#ifndef WAHMM_ALGORITHMS_COMPRESSED_HPP
#define WAHMM_ALGORITHMS_COMPRESSED_HPP

#include "Compressor.hpp"
#include "commons.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

/**
* Compute the compressed forward matrix given a model and a Compressor holding
* the compressed data. Each element approximates the value at the end of the
* block of the uncompressed matrix.
*
* @param m the model
* @param c the compressor holding the data
*/
wahmm::real_t** forward_matrix_compressed(Model& m, Compressor *c){
    wahmm::real_t **logForward;
    size_t numberOfStates = m.mStates.size();
    logForward = new wahmm::real_t*[numberOfStates]; // forward variables

    c->initForward();
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
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
                logForward[j][blockCounter] = sum_logarithms(
                    logForward[j][blockCounter],
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


/**
* Compute the compressed backward matrix given a model and a Compressor holding
* the compressed data. Each element approximates the value at the end of the
* block of the uncompressed matrix.
*
* @param m the model
* @param c the compressor holding the data
*/
wahmm::real_t** backward_matrix_compressed(Model& m, Compressor *c){
    wahmm::real_t **logBackward;
    size_t numberOfStates = m.mStates.size();
    logBackward = new wahmm::real_t*[numberOfStates]; // backward variables

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
                logBackward[i][blockCounter] = sum_logarithms(
                    logBackward[i][blockCounter],
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


/**
* Solve the evaluation problem through a compressed version of the forward
* algorithm.
*
* @param m the model
* @param c compressor holding the data
* @param verbose if true print result informations
* @param silence suppress all output
* @param tofile save results to file
*/
void evaluation_compressed(Model& m, Compressor *c, bool verbose, bool silence,
    bool tofile){
    wahmm::real_t **logForward;
    wahmm::real_t logEvaluation;
    size_t numberOfStates = m.mStates.size();

    if(!silence)
        std::cout << "[>] +++ Compressed Evaluation Problem +++" << std::endl;

    logForward = forward_matrix_compressed(m, c); //initialization and induction
    // termination
    logEvaluation = -infin;
    for(size_t i = 0; i < numberOfStates; i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][c->blocksNumber()-1]);
    }

    // print results
    if(verbose){
        printMatrixSummary(logForward, numberOfStates, c->blocksNumber(),
            "Blocks Forward (log)", false);
    }
    if(!silence)
        std::cout << "[>] log[ P(O|lambda) ]: " << logEvaluation << std::endl;
    if(tofile){
        if(verbose){
            std::cout << "[>] Saving compressed evaluation log probability";
            std::cout << " to file " << PATH_OUT;
            std::cout << "compressed_evaluation_prob ... " << std::flush;
        }
        std::ofstream ofs (PATH_OUT + "compressed_evaluation_prob",
            std::ofstream::out);
        ofs.precision(std::numeric_limits<double>::max_digits10);
        ofs << logEvaluation;
        ofs.close();
        if(verbose)
            std::cout << "done." << std::endl;
    }

    freeMatrix(logForward, numberOfStates);
}


/**
* Solve the decoding problem making use of the compressed version of the
* forward algorithm.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true print result informations
* @param silence suppress all output
* @param tofile save results to file
*/
void decoding_compressed(Model &m, Compressor *c, bool verbose, bool silence,
    bool tofile){
    wahmm::real_t **logViterbi, **statesViterbi;
    size_t numberOfStates = m.mStates.size();
    logViterbi = new wahmm::real_t*[numberOfStates];
    statesViterbi = new wahmm::real_t*[numberOfStates];
    std::vector<size_t> blockLengths;

    if(!silence)
        std::cout << "[>] +++ Compressed Decoding Problem +++" << std::endl;

    c->initForward();
    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        logViterbi[i] = new wahmm::real_t[c->blocksNumber()];
        statesViterbi[i] = new wahmm::real_t[c->blocksNumber()];
        // delta_0(i) = pi_i * e^(E_0(i))
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
                currentSum = logViterbi[i][blockCounter-1] +
                    m.mLogTransitions[i][j];
                if(currentSum > currentMax){
                    currentMax = currentSum;
                    currentState = i;
                }
            }
            // ... e^(E_w(j))
            logViterbi[j][blockCounter] = currentMax +
                compute_e(m, j, c->blockData());
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
        // if currentState == -1, impossible path
        if(currentState >= 0)
            currentState = statesViterbi[currentState][blockCounter];
        else {
            if(!silence)
                std::cerr << "[Warning] Impossible Viterbi path!" << std::endl;
            break;
        }
        viterbiPath.push_front(currentState);
    }

    // print results
    if(verbose){
        printMatrixSummary(logViterbi, numberOfStates, c->blocksNumber(),
            "Block Viterbi (log)", false);
    }
    if(!silence){
        std::cout << "[>] Most likely path: " << std::endl;
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
        std::cout << "[>] log[ P(Q|O,lambda) ]: " << currentMax << std::endl;
    }

    if(tofile){
        if(verbose){
            std::cout << "[>] Saving compressed Viterbi path to file ";
            std::cout << PATH_OUT << "compressed_decoding_path ...";
            std::cout << std::flush;
        }
        std::ofstream ofsPath (PATH_OUT + "compressed_decoding_path",
            std::ofstream::out);
        size_t lenIndex = 0;
        for(auto it = viterbiPath.begin(); it != viterbiPath.end(); it++){
            for(size_t blen = 0; blen < blockLengths[lenIndex]; blen++){
                ofsPath << *it << " ";
            }
            lenIndex++;
        }
        ofsPath.close();
        if(verbose)
            std::cout << "done." << std::endl;
        if(verbose){
            std::cout << "[>] Saving compressed Viterbi log likelihood";
            std::cout << " to file " << PATH_OUT;
            std::cout << "compressed_decoding_prob ... " << std::flush;
        }
        std::ofstream ofsProb (PATH_OUT + "compressed_decoding_prob",
            std::ofstream::out);
        ofsProb.precision(std::numeric_limits<double>::max_digits10);
        ofsProb << currentMax;
        ofsProb.close();
        if(verbose)
            std::cout << "done." << std::endl;
    }

    c->initForward();

    freeMatrix(logViterbi, numberOfStates);
    freeMatrix(statesViterbi, numberOfStates);
}


/**
* Perform one iteration of a compressed version of the Baum-Welch algorithm.
* Both the forward and backward matrix are computed to achieve this.
* The values for K(n_w, j) could be precomputed; to avoid useless computations,
* whenever a new value of n_w is encountered, K(n_w, j) is computed for every
* state and put into a map for greater efficiency.
*
* Parameters reestimation happens by progressively transforming the numbers
* back into the normal space.
*
* @returns the logEvaluation P(O|lambda) of the previous model
*/
wahmm::real_t compressed_baum_welch_iteration(Model& m, Compressor *c,
    wahmm::real_t minSum, wahmm::real_t **logEpsilon, wahmm::real_t *logPi,
    wahmm::real_t **logGamma, wahmm::real_t *logGammaSum,
    wahmm::real_t *logTrDen, wahmm::real_t *average, wahmm::real_t *variance){

    wahmm::real_t logEvaluation; // P(O|lambda)
    wahmm::real_t **logForward; // forward matrix
    wahmm::real_t **logBackward; // backward matrix
    size_t numberOfStates = m.mStates.size();

    // initialization
    for(size_t i = 0; i < numberOfStates; i++){
        for(size_t j = 0; j < numberOfStates; j++)
            logEpsilon[i][j] = -infin;
        logGammaSum[i] = -infin;
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
        average[i] = 0;
        wahmm::real_t v;
        for(int b = 0; b < nBlocks; b++){
            currentBd = c->blockData();
            v = currentBd.s1;
            average[i] += exp(logGamma[i][b]-logGammaSum[i]) * v;
            c->next();
        }
        c->initForward();
        variance[i] = 0;
        for(int b = 0; b < nBlocks; b++){
            currentBd = c->blockData();
            //v can sometimes have some numerical issues caused by HaMMLET code
            v = currentBd.s2 - 2*average[i]*currentBd.s1 +
                currentBd.nw*average[i]*average[i];
            variance[i] += exp(logGamma[i][b]-logGammaSum[i]) * v;
            c->next();
        }
        //variance sometimes gets to 0 due to the numerical issues of v, above;
        // one state always has variance 1, so MIN_VARIANCE should be compared
        // to that
        if(variance[i] < MIN_VARIANCE){
            variance[i] = MIN_VARIANCE;
        }
    }

    //update model
    for(size_t i = 0; i < numberOfStates; i++){
        m.mLogPi[i] = logPi[i];
        for(size_t j = 0; j < numberOfStates; j++){
            m.mLogTransitions[i][j] = logEpsilon[i][j];
        }
        m.mStates[i].updateParameters(average[i],
            sqrt(variance[i]));
    }
    //drop KValues matrix
    m.mKValues.clear();

    freeMatrix(logForward, numberOfStates);
    freeMatrix(logBackward, numberOfStates);

    return logEvaluation;
}


/**
* Solve the training problem by performing more iterations of a compressed
* version of the Baum-Welch algorithm.
* The training continues for a certain number of iterations at maximum or until
* the improvement on the evaluation probability falls below a certain threshold.
* Note that one more iteration than necessary is performed.
*/
void training_compressed(Model& m, Compressor *c, wahmm::real_t thresh,
    size_t maxIterations, bool verbose, bool silence, bool tofile){

    wahmm::real_t **logEpsilon; // eps_t(i,j), accumulator over all t
    // logPi is computed at last
    wahmm::real_t *logPi = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t **logGamma;
    wahmm::real_t *logTrDen = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *logGammaSum = new wahmm::real_t[m.mStates.size()];
    // both average and variance arrays have one element for each state
    wahmm::real_t *average = new wahmm::real_t[m.mStates.size()];
    wahmm::real_t *variance = new wahmm::real_t[m.mStates.size()];

    std::vector<wahmm::real_t> evaluationArray;
    std::vector<wahmm::real_t> improvementArray;

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
    minSum -= 1; // to avoid crash when 0 is saves as -0.0000000001
    wahmm::real_t evaluation=-infin, newEvaluation=-infin;
    wahmm::real_t logImprovement = thresh + 1;
    size_t iter;

    if(!silence)
    std::cout << "[>] +++ Compressed Training Problem +++" << std::endl;

    for(iter = 0; iter < maxIterations && logImprovement > thresh; iter++){
        newEvaluation = compressed_baum_welch_iteration(m, c, minSum,
            logEpsilon, logPi, logGamma,
            logGammaSum, logTrDen, average, variance);
        logImprovement = newEvaluation - evaluation;
        evaluation = newEvaluation;
        improvementArray.push_back(logImprovement);
        evaluationArray.push_back(evaluation);
        if(verbose){
            std::cout << "[>] Iteration: " << iter << std::endl;
            std::cout << "[>] Evaluation improvement (log): ";
            std::cout << logImprovement << std::endl;
            std::cout << "[>] New P(O | lambda): ";
            std::cout << newEvaluation << std::endl;
            m.printModel();
        }
    }
    if(!silence){
        std::cout << "[>] Number of iterations: " << iter << std::endl;
        m.printModel();
    }

    if(tofile){
        if(verbose){
            std::cout << "[>] Saving trained model to file " << PATH_OUT;
            std::cout << "training_model ... " << std::flush;
        }
        std::ofstream modelFileOutput(PATH_OUT + "compressed_training_model");
        std::ofstream modelImprovementOutput(PATH_OUT + "compressed_training_improvement")
        std::ofstream modelLikelihoodOutput(PATH_OUT + "compressed_training_likelihood")
        if(modelFileOutput.is_open()){
            modelFileOutput << m;
        }
        if(verbose){
            std::cout << "number of iterations + others ..." << std::flush;
        }
        if(modelFileOutput.is_open()){
            modelFileOutput << iter;
        }
        for (const auto &e : improvementArray) modelImprovementOutput << e << " ";
        for (const auto &e : evaluationArray) modelLikelihoodOutput << e << " ";
        modelFileOutput.close();
        modelImprovementOutput.close();
        modelLikelihoodOutput.close();
        if(verbose)
            std::cout << "done." << std::endl;
    }

    // free variables
    freeMatrix(logEpsilon, m.mStates.size());
    freeMatrix(logGamma, m.mStates.size());
    delete[] logPi;
    delete[] logGammaSum;
    delete[] average;
    delete[] variance;
}


#endif
