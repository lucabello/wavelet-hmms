#ifndef WAHMM_STATE_HPP
#define WAHMM_STATE_HPP

#include "commons.hpp"

/**
* This class represents the State of a hidden Markov model; it is characterized
* by a continuous Normal emission probability distribution function.
*/
class State {
    static size_t idCounter;
    /** Numerical ID of the state */
    size_t mId;
    /** Name of the state */
    std::string mName;
    /** Mean of the Gaussian associated to the state */
    wahmm::real_t mMean;
    /** Standard deviation of the Gaussian associated to the state */
    wahmm::real_t mStdDev;
    wahmm::real_t mLogStdDev; // precomputed for efficiency reasons
public:
    /** Constructor with default name */
    State(wahmm::real_t mean, wahmm::real_t stdDev);
    /** Constructor with given name */
    State(wahmm::real_t mean, wahmm::real_t stdDev, std::string name);
    size_t id();
    std::string name();
    wahmm::real_t mean();
    wahmm::real_t stdDev();
    wahmm::real_t logStdDev();
    /** The log probability of sampling x from the distribution */
    wahmm::real_t logPdf(wahmm::real_t x);
    void updateParameters(wahmm::real_t mean, wahmm::real_t stdDev);
};

size_t State::idCounter = 0;
const wahmm::real_t log_sqrt2pi = 0.9189385332046727417803297364056176398613974;

State::State(wahmm::real_t mean, wahmm::real_t stdDev) : mMean(mean),
    mStdDev(stdDev),
    mLogStdDev(log(stdDev)),
    mName("State "+std::to_string(idCounter)),
    mId(idCounter){
        idCounter++;
    }

State::State(wahmm::real_t mean, wahmm::real_t stdDev, std::string name) : mMean(mean),
    mStdDev(stdDev),
    mLogStdDev(log(stdDev)),
    mName(name),
    mId(idCounter){
        idCounter++;
    }

size_t State::id(){
    return mId;
}

std::string State::name(){
    return mName;
}

wahmm::real_t State::mean(){
    return mMean;
}

wahmm::real_t State::stdDev(){
    return mStdDev;
}

wahmm::real_t State::logStdDev(){
    return mLogStdDev;
}

wahmm::real_t State::logPdf(wahmm::real_t x){
    return (- log_sqrt2pi - mLogStdDev - 0.5*pow((x - mMean) / mStdDev, 2) );
}

void State::updateParameters(wahmm::real_t mean, wahmm::real_t stdDev){
    mMean = mean;
    mStdDev = stdDev;
    mLogStdDev = log(stdDev);
}

#endif
