#include "NormalDistribution.hpp"
#include "includes.hpp" // from HaMMLET

/**
* This class represents the State of a hidden Markov model; it is characterized
* by a continuous Normal emission probability distribution function.
*/
class State {
    /** Name of the state */
    std::string mName;
    NormalDistribution mDistribution;
public:
    /** Constructor with default name */
    State(NormalDistribution distribution);
    /** Constructor with given name */
    State(NormalDistribution distribution, std::string& name);
    std::string name();
    NormalDistribution distribution();
};

State::State(NormalDistribution distribution) : mDistribution(distribution),
    mName("Default Name"){}

State::State(NormalDistribution distribution, std::string& name) :
    mDistribution(distribution), mName(name){}

std::string State::name(){
    return mName;
}

NormalDistribution State::distribution(){
    return mDistribution;
}
