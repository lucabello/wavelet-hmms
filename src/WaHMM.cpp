#include "parser.hpp"
#include "commons.hpp"
#include "State.hpp"
#include "Model.hpp"
#include "algorithms.hpp"
#include "Compressor.hpp"
#include "algorithms_compressed.hpp"
#include <stdio.h>

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
    bool binary = false;
    bool compressed = false;
    bool verbose = false;
    Compressor *compressor;
    FILE *finObs, *finPath;

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
    if(result.count("binary"))
        binary = true;
    if(result.count("evaluation"))
        evaluation = true;
    if(result.count("decoding"))
        decoding = true;
    if(result.count("training"))
        training = true;
    if(result.count("compressed"))
        compressed = true;
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
    if(fileObs.empty()){
        std::cerr << "[Error] Input file for observations not specified" << std::endl;
        return -1;
    }
    if(!compressed){
        if(!binary){
            wahmm::real_t number;
            // efficient file reading in C style
            // Read the file with input observations
            finObs = fopen(fileObs.c_str(), "r");
            if(finObs != NULL){
                while(fscanf(finObs, "%lf", &number) != EOF)
                    observations.push_back(number);
            } else {
                std::cerr << "Cannot read file " + fileObs + " !" << std::endl;
                return -1;
            }
            fclose(finObs);
        }
        else {
            finObs = fopen ( fileObs.c_str() , "rb" );
            if (finObs==NULL){
                std::cerr << "Cannot read file " + fileObs + " !" << std::endl;
                return -1;
            }
            double n;
            // read one number
            while(fread(&n,1,sizeof(double),finObs) == sizeof(double))
                observations.push_back((wahmm::real_t)n);
            // terminate
            fclose (finObs);
        }
        // // not efficient file reading in C++ style
        // // Read the file with input observations
        // std::ifstream obsFileInput(fileObs);
        // if(obsFileInput.is_open()){
        //     wahmm::real_t number;
        //     while(obsFileInput >> number){
        //         observations.push_back(number);
        //     }
        // } else {
        //     std::cerr << "Cannot read file " + fileObs + " !" << std::endl;
        //     return -1;
        // }
        // obsFileInput.close();

        // std::cout << "Read " << observations.size() << " observations from file." << std::endl;
    }
    else {
        compressor = new Compressor(fileObs, binary);
    }
    // // Read the file with input state path
    if(filePath.empty()){
        std::cerr << "[Warning] Input file for generating path not specified" << std::endl;
    }
        else {
        wahmm::real_t number;
        finPath = fopen(filePath.c_str(), "r");
        if(finPath != NULL){
            while(fscanf(finPath, "%lf", &number) != EOF)
                statePath.push_back(number);
        } else {
            std::cerr << "Cannot read file " + filePath + " !" << std::endl;
            return -1;
        }
        fclose(finPath);
    }

    model.printModel();

    if(!compressed){
        std::cout << "[>] Starting algorithms" << std::endl;
        if(evaluation)
            evaluation_problem(model, observations, verbose);
        if(decoding)
            decoding_problem(model, observations, verbose);
        if(training)
            training_problem_wrapper(model, observations, 1e-9, 100, verbose);
    }
    else {
        std::cout << "[>] Starting compressed algorithms" << std::endl;
        if(evaluation)
            evaluation_compressed(model, compressor, verbose);
        if(decoding)
            decoding_compressed(model, compressor, verbose);
        if(training)
            training_compressed_wrapper(model, compressor, 1e-9, 100, verbose);
    }

    if(result.count("export")){
        std::ofstream modelFileOutput(fileModelOut);
        if(modelFileOutput.is_open()){
            modelFileOutput << model;
        }
        modelFileOutput.close();
    }


    return 0;
}
