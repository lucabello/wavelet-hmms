#include "commons.hpp"
#include "Compressor.hpp"
#include "Model.hpp"
#include "utilities.hpp"
#include "algorithms.hpp"

int main(int argc, const char* argv[]){
    cout.precision(8);
    cout << std::scientific; // print numbers with scientific notation

    std::string obsFile("data/observations");
    std::string pathFile("data/path");
    vector<wahmm::real_t> observations;
    vector<size_t> statePath;
    Model model;
    bool verbose = true, verboseForward = true, verboseViterbi = true;

    // Define the model
    std::vector<State> states{State(0, 1, "State 0"), State(10, 1, "State 1")};
    wahmm::real_t **relativeTransitions;
    relativeTransitions = new wahmm::real_t*[2];
    relativeTransitions[0] = new wahmm::real_t[2];
    relativeTransitions[1] = new wahmm::real_t[2];
    relativeTransitions[0][0] = 1;
    relativeTransitions[0][1] = 1;
    relativeTransitions[1][0] = 1;
    relativeTransitions[1][1] = 1;
    vector<wahmm::real_t> initDist{0.0, -inf};
    model = Model(states, relativeTransitions, initDist);

    model.printModel();

    // Read the file with input observations
    std::ifstream obsFinput(obsFile);
    if(obsFinput.is_open()){
        wahmm::real_t number;
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
        wahmm::real_t number;
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
    Model estimate_model;
    std::vector<State> estates{State(0, 1, "State 0"), State(10, 1, "State 1")};
    wahmm::real_t **erelativeTransitions;
    erelativeTransitions = new wahmm::real_t*[2];
    erelativeTransitions[0] = new wahmm::real_t[2];
    erelativeTransitions[1] = new wahmm::real_t[2];
    erelativeTransitions[0][0] = 1;
    erelativeTransitions[0][1] = 1;
    erelativeTransitions[1][0] = 1;
    erelativeTransitions[1][1] = 1;
    vector<wahmm::real_t> einitDist{0.0, -inf};
    estimate_model = Model(estates, erelativeTransitions, einitDist);

    // Try to solve the third problem
    evaluation_problem(estimate_model, observations, verboseForward);
    cout << "+++++ Training Problem +++++" << endl;
    training_problem_wrapper(estimate_model, observations, 1e-9, 100);

    /*
    estimate_model.printModel();
    Memorizing the file two times is too much and causes segmentation fault.
    Choose one between the uncompressed or compressed version when executing
    the program.
    */
    std::string filename;
    //compressor = Compressor(filename);
    //compressor.printAllBlocks(); // debug check

    return 0;
}
