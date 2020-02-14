#include "includes.hpp"
#include "Compressor.hpp"
#include "Model.hpp"
#include "utilities.hpp"
#include <list>
using std::list;

void evaluation_prob(Model& m, vector<real_t>& obs, bool verbose);
void decoding_prob(Model& m, vector<real_t>& obs, bool verbose);

int main(int argc, const char* argv[]){
    cout << std::scientific; // print numbers with scientific notation

    std::string obsFile("data/observations");
    std::string pathFile("data/path");
    vector<real_t> observations;
    vector<size_t> statePath;
    Model model;
    bool verbose = true, verboseForward = true, verboseViterbi = true;

    if(verbose)
        cout << "Trying to define model...";

    // Define the model
    State s0(NormalDistribution(0, 5), "State 0");
    State s1(NormalDistribution(10, 5), "State 1");
    vector<State> states;
    states.push_back(s0);
    states.push_back(s1);
    real_t **logTrans;
    logTrans = new real_t*[2];
    logTrans[0] = new real_t[2];
    logTrans[1] = new real_t[2];
    logTrans[0][0] = log(0.5);
    logTrans[0][1] = log(0.5);
    logTrans[1][0] = log(0.5);
    logTrans[1][1] = log(0.5);
    vector<real_t> initDist;
    initDist.push_back(0.0);
    initDist.push_back(-inf);
    model = Model(states, logTrans, initDist);

    if(verbose){
        cout << " success!" << endl;
        model.printModel();
        cout << "Trying to read files...";
    }

    // Read the file with input observations
    ifstream obsFinput(obsFile);
    if(obsFinput.is_open()){
        real_t number;
        while(obsFinput >> number){
            observations.push_back(number);
        }
    } else {
        cerr << "Cannot read file " + obsFile + " !" << endl;
        return -1;
    }
    obsFinput.close();
    // Read the file with input state path
    ifstream pathFinput(pathFile);
    if(pathFinput.is_open()){
        real_t number;
        while(pathFinput >> number){
            statePath.push_back(number);
        }
    } else {
        cerr << "Cannot read file " + pathFile + " !" << endl;
        return -1;
    }
    pathFinput.close();

    if(verbose)
        cout << " success!" << endl;

    // Try to solve the first prolem
    evaluation_prob(model, observations, verboseForward);
    // Try to solve the second problem
    decoding_prob(model, observations, verboseViterbi);

    /*
    Memorizing the file two times is too much and causes segmentation fault.
    Choose one between the uncompressed or compressed version when executing
    the program.
    */
    //Compressor compressor(filename);
    //compressor.printAllBlocks(); // debug check

    return 0;
}

/**
* Solve the evaluation problem through the forward algorithm.
* If verbose it prints the forward matrix.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true prints result informations
*/
void evaluation_prob(Model& m, vector<real_t>& obs, bool verbose){
    real_t **logForward;
    logForward = new real_t*[m.mStates.size()]; // forward variables

    if(verbose)
        cout << "+++++ Evaluation Problem +++++" << endl;

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

    // termination
    real_t logEvaluation = -inf;
    for(int i = 0; i < m.mStates.size(); i++){
        logEvaluation = sum_logarithms(logEvaluation,
            logForward[i][obs.size()-1]);
    }

    // print results
    if(verbose){
        printMatrixSummary(logForward, m.mStates.size(), obs.size(),
            "logForward", false);
        cout << "ObservationSequence LogProbability: " << logEvaluation << endl;
    }

    for(int i = 0; i < m.mStates.size(); i++)
        delete[] logForward[i];
    delete[] logForward;
}


/**
* Solve the evaluation problem through the forward algorithm.
* Currently write the obtained path in results/wahmm_viterbi.
* If verbose it prints the Viterbi matrix.
*
* @param m the model
* @param obs the observation sequence
* @param verbose if true prints result informations
*/
void decoding_prob(Model &m, vector<real_t>& obs, bool verbose){
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
        cout << "Path logProbability: " << currentMax << endl;
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
