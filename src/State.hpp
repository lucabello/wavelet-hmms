#include "NormalDistribution.hpp"
#include "includes.hpp" // from HaMMLETmId

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
    NormalDistribution mDistribution;
public:
    /** Constructor with default name */
    State(NormalDistribution distribution);
    /** Constructor with given name */
    State(NormalDistribution distribution, std::string name);
    std::string name();
    NormalDistribution distribution();
    size_t id();
    void setDistribution(NormalDistribution d);
};

size_t State::idCounter = 0;

State::State(NormalDistribution distribution) : mDistribution(distribution),
    mName("Default Name"), mId(idCounter++){}

State::State(NormalDistribution distribution, std::string name) :
    mDistribution(distribution), mName(name), mId(idCounter++){}

size_t State::id(){
    return mId;
}

std::string State::name(){
    return mName;
}

NormalDistribution State::distribution(){
    return mDistribution;
}

void State::setDistribution(NormalDistribution d){
    mDistribution = d;
}
