#include "includes.hpp" //from HaMMLET

class State {
    /** Name of the state */
    std::string mName;
    /** Mean of the Normal distribution */
    real_t mMean;
    /** Variance of the Normal distribution */
    real_t mVariance;
public:
    /** Constructor with default name */
    State(real_t& mean, real_t& variance);
    /** Constructor with given name */
    State(real_t& mean, real_t& variance, std::string& name);
    std::string name();
    real_t mean();
    real_t variance();
    void setMean(real_t mean);
    void setVariance(real_t variance);
};

State::State(real_t& mean, real_t& variance) : mMean(mean),
    mVariance(variance){}

State::State(real_t& mean, real_t& variance, std::string& name) : mMean(mean),
    mVariance(variance), mName(name){}

std::string State::name(){
    return mName;
}

real_t State::mean(){
    return mMean;
}

real_t State::variance(){
    return mVariance;
}

void State::setMean(real_t mean){
    mMean = mean;
}

void State::setVariance(real_t variance){
    mVariance = variance;
}
