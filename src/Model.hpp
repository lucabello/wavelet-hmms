#ifndef WAHMM_MODEL_HPP
#define WAHMM_MODEL_HPP

#include "State.hpp"
#include "commons.hpp"

/*
This class represents a hidden Markov model for the specific scope
and context of the thesis (Gaussian states, continuous observations, etc).
The member variables are all public for efficiency reasons.
*/
class Model {
public:
    /** Collection of the states of the model */
    std::vector<State> mStates;
    /** Matrix of log probabilities of transitions between states */
    wahmm::real_t **mLogTransitions;
    /** Logarithm of the initial state probability distribution */
    std::vector<wahmm::real_t> mLogPi;
    /** Each entry is < n_w , K(n_w,j) for each state > */
    std::map<size_t, std::vector<wahmm::real_t>> mKValues;
    Model();
    Model(const Model& that);
    Model(std::vector<State>& states, std::vector<wahmm::real_t>&relativeTr,
        std::vector<wahmm::real_t>& relativePi);
    /** Print the model in a readable format, with classic probabilities */
    void printModel();
    /** Sort the model with ascending states order */
    void sortModel();
    // Useful operators for model input/output
    friend ostream& operator<<(ostream& os, const Model& m);
    friend istream& operator>>(istream& is, Model& m);
};

Model::Model(){}

Model::Model(const Model& that){
    mStates = that.mStates;
    for(int i = 0; i < mStates.size(); i++){
        mLogTransitions[i] = new wahmm::real_t[mStates.size()];
        for(int j = 0; j < mStates.size(); j++){
            mLogTransitions[i][j] = that.mLogTransitions[i][j];
        }
    }
    mLogPi = that.mLogPi;
}

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
    std::cout << "[>] Model information:" << std::endl;
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

void Model::sortModel(){
    for(size_t i = 0; i < mStates.size(); i++){
        for(size_t j = 0; j < mStates.size()-1; j++){
            if(mStates[j].mean() > mStates[j+1].mean()){
                State tmpState = mStates[j];
                mStates[j] = mStates[j+1];
                mStates[j+1] = tmpState;
                wahmm::real_t *tmpTr = mLogTransitions[j];
                mLogTransitions[j] = mLogTransitions[j+1];
                mLogTransitions[j+1] = tmpTr;
                wahmm::real_t tmpInit = mLogPi[j];
                mLogPi[j] = mLogPi[j+1];
                mLogPi[j+1] = tmpInit;
            }
        }
    }
}

ostream& operator<<(ostream& os, const Model& m){
    os.precision(100);
    os << m.mStates.size() << " ";
    for(State s : m.mStates)
        os << s.mean() << " " << s.stdDev() << " ";
    for(size_t i = 0; i < m.mStates.size(); i++)
        for(size_t j = 0; j < m.mStates.size(); j++)
            os << m.mLogTransitions[i][j] << " ";
    for(size_t i = 0; i < m.mStates.size(); i++)
        os << m.mLogPi[i] << " ";
    return os;
}

istream& operator>>(istream& is, Model& m){
    size_t nStates;
    wahmm::real_t inMean, inStdDev;
    is >> nStates;
    for(size_t n = 0; n < nStates; n++){
        is >> inMean >> inStdDev;
        m.mStates.push_back(State(inMean, inStdDev));
    }
    wahmm::real_t inTr;
    std::string inTrString;
    m.mLogTransitions = new wahmm::real_t*[nStates];
    for(size_t i = 0; i < nStates; i++){
        m.mLogTransitions[i] = new wahmm::real_t[nStates];
        for(size_t j = 0; j < nStates; j++){
            is >> inTrString;
            if(inTrString == "-inf")
                inTr = -infin;
            else
                inTr = std::stod(inTrString);
            m.mLogTransitions[i][j] = inTr;
        }
    }
    wahmm::real_t inPi;
    std::string inPiString;
    for(size_t i = 0; i < nStates; i++){
        is >> inPiString;
        if(inPiString == "-inf")
            inPi = -infin;
        else
            inPi = std::stod(inPiString);
        m.mLogPi.push_back(inPi);
    }
    return is;
}

#endif
