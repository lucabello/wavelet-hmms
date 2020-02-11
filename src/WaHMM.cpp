#include "includes.hpp"
#include "Compressor.hpp"
#include "Model.hpp"

int main(int argc, const char* argv[]){
    std::string filename("data");
    vector<real_t> observations;
    vector<size_t> statePath;
    Model model;

    // Read the file with input data
    ifstream finput(filename);
    if(finput.is_open()){
        real_t number;
        while(finput >> number){
            observations.push_back(number);
        }
    } else {
        cerr << "Cannot read file " + filename + " !" << endl;
        return -1;
    }

    cout << observations.size() << endl;
    finput.close();

    /*
    Memorizing the file two times is too much and causes segmentation fault.
    Choose one between the uncompressed or compressed version when executing
    the program.
    */
    //Compressor compressor(filename);
    //compressor.printAllBlocks(); // debug check

    return 0;
}

/** Given (a = log(x) and b = log(y), returns log(x+y))
real_t sum_logarithms(real_t& a, real_t& b){
    //TODO : Write the correct implementation!
    // Check out pomegranate for it...
    return a + b;
}

real_t evaluation_prob(Model& m, vector<real_t>& obs, vector<real_t>& path,
    bool verbose){
    real_t **logForward;
    logForward = new real_t*[m.mStates.size()];

    // initialization
    for(int i = 0; i < m.mStates.size(); i++){
        logForward[i] = new real_t[obs.size()];
        logForward[i][0] = m.mInitialDistribution[i] + m.mStates[path[i]]
            .distribution().pdf(obs[0]);
    }

    // induction
    for(int o = 1; j < obs.size(); j++){ // observations
        // TODO: Initialize the logForward[j][o] = -Inf for every j
        for(int j = 0; j < m.mStates.size(); j++){ // arriving state
            for(int i = 0; i < m.mStates.size(); i++){ // starting state
                // alpha_t+1(j) = \sum_{i=0}^N alpha_t(i)a_{ij} ...
                logForward[j][o] = sum_logarithms(logForward[j][o],
                    logForward[i][o-1] + m.mLogTransitions[i][j]);
            }
            // ... b_{j}(O_{t+1})
            logForward[j][o] += m.mStates[path[o]].distribution.pdf(obs[o]);
        }
    }

    // termination
    real_t logEvaluation = 0; // TODO: result should be initialized at -Inf
    for(int i = 0; i < m.mStates.size(); i++){
        logEvaluation = sum_logarithms(result, logForward[i])
    }

    if(verbose){
        cout << "logForward: " << endl;
        for(int j = 0; j < m.mStates.size(); j++){
            for(int o = 0; o < obs.size(); o++){
                cout << logForward[j][o] << " ";
            }
            cout << endl;
        }
    }

    return logEvaluation;
}
