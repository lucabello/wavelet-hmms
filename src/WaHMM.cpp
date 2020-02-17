#include "includes.hpp"
#include "Compressor.hpp"
#include "Model.hpp"
#include "utilities.hpp"
#include "algorithms.hpp"

int main(int argc, const char* argv[]){
    cout.precision(8);
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
    //evaluation_problem(model, observations, verboseForward);
    // Try to solve the second problem
    //decoding_problem(model, observations, verboseViterbi);

    // Define a starting estimate of the model
    State es0(NormalDistribution(0, 5), "State 0");
    State es1(NormalDistribution(30, 10), "State 1");
    vector<State> estates;
    estates.push_back(es0);
    estates.push_back(es1);
    real_t **elogTrans;
    elogTrans = new real_t*[2];
    elogTrans[0] = new real_t[2];
    elogTrans[1] = new real_t[2];
    elogTrans[0][0] = log(0.3);
    elogTrans[0][1] = log(0.7);
    elogTrans[1][0] = log(0.7);
    elogTrans[1][1] = log(0.3);
    vector<real_t> einitDist;
    einitDist.push_back(0.0);
    einitDist.push_back(-inf);
    Model estimate_model(estates, elogTrans, einitDist);

    // Try to solve the third problem
    cout << "+++++ Training Problem +++++" << endl;
    training_problem_wrapper(estimate_model, observations, 1e-9, 100);
    estimate_model.printModel();

    /*
    Memorizing the file two times is too much and causes segmentation fault.
    Choose one between the uncompressed or compressed version when executing
    the program.
    */
    //Compressor compressor(filename);
    //compressor.printAllBlocks(); // debug check

    return 0;
}
