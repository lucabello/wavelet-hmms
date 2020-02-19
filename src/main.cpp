#include "parser.hpp"
#include "commons.hpp"
#include "State.hpp"
#include "Model.hpp"

int main(int argc, const char* argv[]){
    // std::string filename("data");
    // Compressor comp(filename);
    // comp.printAllBlocks();
    auto result = parse(argc, argv);
    auto arguments = result.arguments();
    std::cout << "Saw " << arguments.size() << " arguments" << std::endl;

    std::vector<State> states;
    std::vector<wahmm::real_t> logTrans;
    std::vector<wahmm::real_t> logPi;
    std::string fileObs, filePath;
    bool evaluation = false, decoding = false, training = false;
    bool verbose = false;

    if(result["state"].count()){
        std::vector<double> stateParams = result["state"].as<std::vector<double>>();
        wahmm::real_t mean, stdDev;
        for(std::size_t i = 0; i < result.count("state"); i++){
            mean = stateParams[i*2];
            stdDev = stateParams[i*2 + 1];
            states.push_back(State(mean, stdDev));
        }
    }
    
    for(auto arg : arguments){
        if(arg.key() == "transitions"){
            for(double d : arg.value())
                logTrans.push_back((wahmm::real_t)d);
        }
        else if(arg.key() == "initial"){
            for(double d : arg.value())
                logPi.push_back((wahmm::real_t)d);
        }
        else if(arg.key() == "obs"){
            fileObs = arg.value();
        }
        else if(arg.key() == "path"){
            filePath = arg.value();
        }
    }
    if(result.count("evaluation"))
        evaluation = true;
    if(result.count("decoding"))
        decoding = true;
    if(result.count("training"))
        training = true;
    if(result.count("verbose"))
        verbose = true;

    Model m(states, logTrans, logPi);
    m.printModel();


    for(auto s : states){
        std::cout << s.mean() << " " << s.stdDev() << std::endl;
    }

    return 0;
}
