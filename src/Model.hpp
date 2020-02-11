#include "State.hpp"
#include "includes.hpp"

/*
This class should represent a Hidden Markov Model for the specific scope
and context of the thesis (Normal states, continuous observations, etc).
Another bigger class will include both this and the Compressor, so that it can
solve problem 3 without the Model (even returning a new Model) and it can ask
for the Model to solve problem 1 and 2. It would be also able ot use the
solution of problem 3 to generate a Model on which apply problem 1 and 2.
*/
class Model {
    vector<State> mStates;
    real_t **mLogTransitions;
    vector<real_t> mInitialDistribution;
public:
    Model(vector<State>& states, real_t **logTransitions,
        vector<real_t>& initialDistribution){}
    void printModel();
};

Model::Model(vector<State>& states, real_t **logTransitions,
    vector<real_t>& initialDistribution){
    mStates = vector<State>(states);
    mLogTransitions = new real_t*[states.size()];
    for(int i = 0; i < states.size(); i++)
        mLogTransitions = new real_t[states.size()];
    for(int i = 0; i < states.size(); i++)
        for(int j = 0; j < states.size(); j++)
            mLogTransitions[i][j] = logTransitions[i][j];
    mInitialDistribution = vector<real_t>(initialDistribution);
}

Model::printModel(){
    for(State s : mStates){
        cout << s.name();
        cout << " | Mean: " << s.mean();
        cout << " | StdDev: " << s.stdDev();
        cout << endl;
    }
    cout << "++++++++++" << endl;
    cout << "Transitions: " << endl;
    for(int i = 0; i < mStates.size(); i++){
        for(int j = 0; j < mStates.size(); j++)
            cout << mLogTransitions[i][j] << " ";
        cout << endl;
    }
    cout << "++++++++++" << endl;
    cout << "Initial Distribution: " << endl;
    for(int i = 0; i < mStates.size(); i++)
        cout << mInitialDistribution[i] << " ";
    cout << endl;
    cout << "++++++++++" << endl;
}
