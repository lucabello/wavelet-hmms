#include "parser.hpp"
#include "commons.hpp"
#include "State.hpp"
#include "Model.hpp"
#include "algorithms.hpp"

int main(int argc, const char* argv[]){
    // std::string filename("data");
    // Compressor comp(filename);
    // comp.printAllBlocks();
    //std::cout.precision(8);
    //std::cout << std::scientific; // print numbers with scientific notation
    auto result = parse(argc, argv);

    Model model;
    std::vector<State> states;
    std::vector<wahmm::real_t> relTrans;
    std::vector<wahmm::real_t> relPi;
    std::vector<wahmm::real_t> observations;
    std::vector<wahmm::real_t> statePath;
    std::string fileObs, filePath, fileModelIn, fileModelOut;
    bool evaluation = false, decoding = false, training = false;
    bool verbose = false;

    // parsing arguments from command line
    if(result.count("import")){ // read the model from a file
        fileModelIn = result["import"].as<std::string>();
        // Read the file with input observations
        std::ifstream modelFileInput(fileModelIn);
        if(modelFileInput.is_open()){
            modelFileInput >> model;
        }
        else {
            std::cerr << "Cannot read file " + fileModelIn + " !" << std::endl;
            return -1;
        }
        modelFileInput.close();
    }
    else { // read the model from command line
        if(result.count("state")){
            std::vector<double> stateParams = result["state"].as<std::vector<double>>();
            wahmm::real_t mean, stdDev;
            for(std::size_t i = 0; i < result.count("state"); i++){
                mean = stateParams[i*2];
                stdDev = stateParams[i*2 + 1];
                states.push_back(State(mean, stdDev));
            }
        }
        if(result.count("transitions")){
            std::vector<double> transParams = result["transitions"].as<std::vector<double>>();
            for(double d : transParams)
                relTrans.push_back((wahmm::real_t)d);
        }
        if(result.count("initial")){
            std::vector<double> logParams = result["initial"].as<std::vector<double>>();
            for(double d : logParams)
                relPi.push_back((wahmm::real_t)d);
        }
        model = Model(states, relTrans, relPi);
    }
    if(result.count("export")){
        fileModelOut = result["export"].as<std::string>();
    }
    if(result.count("obs")){
        fileObs = result["obs"].as<std::string>();
    }
    if(result.count("path")){
        filePath = result["path"].as<std::string>();
    }
    if(result.count("evaluation"))
        evaluation = true;
    if(result.count("decoding"))
        decoding = true;
    if(result.count("training"))
        training = true;
    if(result.count("verbose"))
        verbose = true;

    // some input checks
    if(states.size() != relPi.size()){
        std::cerr << "[Error] Wrong initial distribution size" << std::endl;
        return -1;
    }
    if(states.size()*states.size() != relTrans.size()){
        std::cerr << "[Error] Wrong number of transition probabilities" << std::endl;
        return -1;
    }

    // Read the file with input observations
    std::ifstream obsFileInput(fileObs);
    if(obsFileInput.is_open()){
        wahmm::real_t number;
        while(obsFileInput >> number){
            observations.push_back(number);
        }
    } else {
        std::cerr << "Cannot read file " + fileObs + " !" << std::endl;
        return -1;
    }
    obsFileInput.close();
    // Read the file with input state path
    std::ifstream pathFileInput(filePath);
    if(pathFileInput.is_open()){
        wahmm::real_t number;
        while(pathFileInput >> number){
            statePath.push_back(number);
        }
    } else {
        std::cerr << "Cannot read file " + filePath + " !" << std::endl;
        return -1;
    }
    pathFileInput.close();

    model.printModel();

    if(evaluation)
        evaluation_problem(model, observations, verbose);
    if(decoding)
        decoding_problem(model, observations, verbose);
    if(training)
        training_problem_wrapper(model, observations, 1e-9, 100, verbose);

    if(result.count("export")){
        std::ofstream modelFileOutput(fileModelOut);
        if(modelFileOutput.is_open()){
            modelFileOutput << model;
        }
        modelFileOutput.close();
    }


    return 0;
}
