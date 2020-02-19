#ifndef WAHMM_MODEL_HPP
#define WAHMM_MODEL_HPP

#include "State.hpp"
#include "commons.hpp"

/*
This class should represent a Hidden Markov Model for the specific scope
and context of the thesis (Normal states, continuous observations, etc).
Another bigger class will include both this and the Compressor, so that it can
solve problem 3 without the Model (even returning a new Model) and it can ask
for the Model to solve problem 1 and 2. It would be also able ot use the
solution of problem 3 to generate a Model on which apply problem 1 and 2.
*/
class Model {
public:
    std::vector<State> mStates;
    wahmm::real_t **mLogTransitions;
    std::vector<wahmm::real_t> mLogPi;
    Model();
    Model(const Model& that) = delete;
    Model(std::vector<State>& states, std::vector<wahmm::real_t>&relativeTr,
        std::vector<wahmm::real_t>& relativePi);
    void printModel();
};

Model::Model(){}

Model::Model(std::vector<State>& states, std::vector<wahmm::real_t>& relativeTr,
    std::vector<wahmm::real_t>& relativePi){
    mStates = std::vector<State>(states);
    mLogTransitions = new wahmm::real_t*[states.size()];
    wahmm::real_t rowSum;
    for(int i = 0; i < states.size(); i++){
        mLogTransitions[i] = new wahmm::real_t[states.size()];
        rowSum = 0;
        for(int j = 0; j < states.size(); j++)
            rowSum += relativeTr[i*states.size() + j];
        for(int j = 0; j < states.size(); j++)
            mLogTransitions[i][j] = log(relativeTr[i*states.size() + j]) -
                log(rowSum);
        rowSum = 0;
        for(int j = 0; j < states.size(); j++)
            rowSum += relativePi[j];
        mLogPi.push_back(log(relativePi[i]) - log(rowSum));
    }
}

void Model::printModel(){
    std::cout << "----------" << std::endl;
    for(State s : mStates){
        cout << s.name();
        cout << " | Mean: " << s.mean();
        cout << " | StdDev: " << s.stdDev();
        cout << endl;
    }
    std::cout << "----------" << std::endl;
    std::cout << "Transitions: " << std::endl;
    for(int i = 0; i < mStates.size(); i++){
        for(int j = 0; j < mStates.size(); j++)
            std::cout << exp(mLogTransitions[i][j]) << " ";
        std::cout << std::endl;
    }
    std::cout << "----------" << std::endl;
    std::cout << "Initial Distribution: " << std::endl;
    for(int i = 0; i < mStates.size(); i++)
        std::cout << exp(mLogPi[i]) << " ";
    std::cout << endl;
    std::cout << "----------" << std::endl;
}

#endif
