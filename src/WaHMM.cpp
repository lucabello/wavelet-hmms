#include "parser.hpp"
#include "commons.hpp"
#include "State.hpp"
#include "Model.hpp"
#include "algorithms.hpp"
#include "Compressor.hpp"
#include "algorithms_compressed.hpp"

#include <stdio.h>

int main(int argc, const char* argv[]){
    auto result = parse(argc, argv);

    Model model, estimate;
    std::vector<State> states;
    std::vector<wahmm::real_t> relTrans;
    std::vector<wahmm::real_t> relPi;
    std::vector<wahmm::real_t> observations;
    std::string fileObs, fileModelIn, pathOut;
    bool evaluation = false, decoding = false, training = false;
    bool binary = false, tofile = false;
    bool compressed = false;
    Compressor *compressor;
    FILE *finObs, *finPath;

    // Configure the log level here
    AixLog::Log::init<AixLog::SinkCout>(AixLog::Severity::error);

    // Parsing arguments from command line
    if (result.count("model")) { // read the model from a file
        fileModelIn = result["model"].as<std::string>();
        // Read the file with input observations
        model = Model(fileModelIn);
    }
    else { // read the model from command line
        if (result.count("state")) {
            std::vector<double> stateParams = result["state"]
                .as<std::vector<double>>();
            wahmm::real_t mean, stdDev;
            for (std::size_t i = 0; i < result.count("state"); i++) {
                mean = stateParams[i*2];
                stdDev = stateParams[i*2 + 1];
                states.push_back(State(mean, stdDev));
            }
        }
        if (result.count("transitions")) {
            std::vector<double> transParams = result["transitions"]
                .as<std::vector<double>>();
            for (double d : transParams)
                relTrans.push_back((wahmm::real_t)d);
        }
        if (result.count("initial")) {
            std::vector<double> logParams = result["initial"]
                .as<std::vector<double>>();
            for (double d : logParams)
                relPi.push_back((wahmm::real_t)d);
        }
        model = Model(states, relTrans, relPi);
    }
    if (result.count("estimate")) {
        fileModelIn = result["estimate"].as<std::string>();
        // Read the file with input observations
        estimate = Model(fileModelIn);
    }
    if (result.count("obs")) {
        fileObs = result["obs"].as<std::string>();
    }
    if (result.count("binary"))
        binary = true;
    if (result.count("tofile"))
        tofile = true;
    if (result.count("evaluation"))
        evaluation = true;
    if (result.count("decoding"))
        decoding = true;
    if (result.count("training"))
        training = true;
    if (result.count("compressed"))
        compressed = true;

    // some input checks
    if (states.size() != relPi.size()) {
        LOG(ERROR) << "Wrong initial distribution size" << std::endl;
        return -1;
    }
    if (states.size()*states.size() != relTrans.size()) {
        LOG(ERROR) << "Wrong number of transition probabilities" << std::endl;
        return -1;
    }
    if (fileObs.empty()) {
        LOG(ERROR) << "Input file for observations not specified" << std::endl;
        return -1;
    }

    // read observations
    if (!compressed) {
        if (!binary) {
            LOG(DEBUG) << "Reading observations... " << std::endl;
            wahmm::real_t number;
            // efficient file reading in C style
            // Read the file with input observations
            finObs = fopen(fileObs.c_str(), "r");
            if(finObs != NULL){
                while(fscanf(finObs, "%lf", &number) != EOF)
                    observations.push_back(number);
            } else {
                LOG(ERROR) << "Cannot read file " + fileObs + " !" << std::endl;
                return -1;
            }
            fclose(finObs);
            LOG(DEBUG) << "... done." << std::endl;
        }
        else {
            LOG(DEBUG) << "Reading observations from binary file... " << std::endl;
            finObs = fopen ( fileObs.c_str() , "rb" );
            if (finObs==NULL) {
                LOG(ERROR) << "Cannot read file " + fileObs + " !" << std::endl;
                return -1;
            }
            double n;
            // read one number
            while (fread(&n,1,sizeof(double),finObs) == sizeof(double))
                observations.push_back((wahmm::real_t)n);
            // terminate
            fclose(finObs);
            LOG(DEBUG) << "... done." << std::endl;
        }
    }
    else {
        LOG(DEBUG) << "Creating Compressor... " << std::endl;
        compressor = new Compressor(fileObs, binary);
        LOG(DEBUG) << "... done." << std::endl;
    }

    // execute the actual algorithms
    if(!compressed){
        LOG(INFO) << "Starting standard algorithms." << std::endl;
        if(evaluation)
            evaluation_problem(model, observations);
        if(decoding)
            decoding_problem(model, observations);
        if(training)
            training_problem(estimate, observations, 1e-3, 100);
    }
    else {
        LOG(INFO) << "Starting compressed algorithms" << std::endl;
        if(evaluation)
            evaluation_compressed(model, compressor);
        if(decoding)
            decoding_compressed(model, compressor);
        if(training)
            training_compressed(estimate, compressor, 1e-3, 100);
    }

    return 0;
}
