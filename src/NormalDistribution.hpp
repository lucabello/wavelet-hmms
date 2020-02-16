#include "includes.hpp" // from HaMMLET

class NormalDistribution {
    /** Mean of the Normal distribution */
    real_t mMean;
    /** Standard deviation of the Normal distribution */
    real_t mStdDev;
    /** Log of standard deviation for computational speed */
    real_t mLogStdDev;
public:
    NormalDistribution(real_t mean, real_t stdDev);
    real_t mean();
    real_t stdDev();
    /**
    * Evaluates the Normal probability distribution function at x. The
    * implementation is:
    * log(1/2pi) - log(stdDev) - 0.5((x-mean)/stdDev)^2
    */
    real_t pdf(real_t x);
};

NormalDistribution::NormalDistribution(real_t mean, real_t stdDev):
    mMean(mean), mStdDev(stdDev), mLogStdDev(log(stdDev)){}

real_t NormalDistribution::mean(){
    return mMean;
}

real_t NormalDistribution::stdDev(){
    return mStdDev;
}

real_t NormalDistribution::pdf(real_t x){
    //TODO: write more efficient implementation
    const real_t log_sqrt2pi = 0.9189385332046727417803297364056176398613974;
    return (- log_sqrt2pi - mLogStdDev - 0.5*pow((x - mMean) / mStdDev, 2) );
}
