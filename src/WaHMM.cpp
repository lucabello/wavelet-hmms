#include "includes.hpp"
#include "Compressor.hpp"
#include "Model.hpp"

real_t evaluation_prob(Model& m, vector<real_t>& obs, bool verbose);

int main(int argc, const char* argv[]){
    std::string obsFile("data");
    std::string pathFile("path");
    vector<real_t> observations;
    vector<size_t> statePath;
    Model model;
    bool verbose = true, verboseForward = true;

    if(verbose)
        cout << "Trying to define model...";

    // Define the model
    State s0(NormalDistribution(0, 1), "State 0");
    State s1(NormalDistribution(10, 1), "State 1");
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
    real_t logEval = evaluation_prob(model, observations, verboseForward);

    if(verbose)
        cout << "The final result is P(O|lambda): " << logEval << endl;

    /*
    Memorizing the file two times is too much and causes segmentation fault.
    Choose one between the uncompressed or compressed version when executing
    the program.
    */
    //Compressor compressor(filename);
    //compressor.printAllBlocks(); // debug check

    return 0;
}

/** Given (a = log(x) and b = log(y), returns log(x+y)) */
real_t sum_logarithms(real_t& a, real_t& b){
    /**
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	*/
    //TODO : Write the correct implementation!
    if(a == inf || b == inf)
        return inf;
    if(a == -inf)
        return b;
    if(b == -inf)
        return a;
    if(a > b)
        return a + log(exp(b-a) + 1);
    return b + log(exp(a-b) + 1);
}

real_t evaluation_prob(Model& m, vector<real_t>& obs, bool verbose){
    real_t **logForward;
    logForward = new real_t*[m.mStates.size()];

    // initialization
    for(int i = 0; i < m.mStates.size(); i++){
        logForward[i] = new real_t[obs.size()];
        logForward[i][0] = m.mInitialDistribution[i] + m.mStates[i]
            .distribution().pdf(obs[0]);
    }

    real_t newContrib;
    // induction
    for(int o = 1; o < obs.size(); o++){ // observations
        for(int j = 0; j < m.mStates.size(); j++)
            logForward[j][o] = -inf;
        // TODO: Initialize the logForward[j][o] = -Inf for every j
        for(int j = 0; j < m.mStates.size(); j++){ // arriving state
            for(int i = 0; i < m.mStates.size(); i++){ // starting state
                // alpha_t+1(j) = \sum_{i=0}^N alpha_t(i)a_{ij} ...
                newContrib = logForward[i][o-1] + m.mLogTransitions[i][j];
                logForward[j][o] = sum_logarithms(logForward[j][o],
                    newContrib);
            }
            // ... b_{j}(O_{t+1})
            logForward[j][o] += m.mStates[j].distribution().pdf(obs[o]);
        }
    }

    // termination
    real_t logEvaluation = -inf;// TODO: result should be initialized at -Inf
    for(int i = 0; i < m.mStates.size(); i++){
        logEvaluation = sum_logarithms(logEvaluation, logForward[i][obs.size()-1]);
    }

    if(verbose){
        cout << "logForward: " << endl;
        if(obs.size() <= 6){
            for(int o = 0; o < obs.size(); o++){
                cout << "[";
                for(int j = 0; j < m.mStates.size(); j++){
                    cout << logForward[j][o] << " ";
                }
                cout << "]" << endl;
            }
        }
        else{
            cout << "[" << logForward[0][0] << " ";
            cout << logForward[1][0] << "]" << endl;
            cout << "[" << logForward[0][1] << " ";
            cout << logForward[1][1] << "]" << endl;
            cout << "..." << endl;
            cout << "[" << logForward[0][obs.size()-3] << " ";
            cout << logForward[1][obs.size()-3] << "]" << endl;
            cout << "[" << logForward[0][obs.size()-2] << " ";
            cout << logForward[1][obs.size()-2] << "]" << endl;
            cout << "[" << logForward[0][obs.size()-1] << " ";
            cout << logForward[1][obs.size()-1] << "]" << endl;
        }
        if(logForward[0][obs.size()-1] == logForward[1][obs.size()-1]){
            cout << "ATTENTION! last items are equal" << endl;
        }
    }

    return logEvaluation;
}
