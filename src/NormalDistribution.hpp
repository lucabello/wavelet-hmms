#include "includes.hpp" // from HaMMLET

class NormalDistribution {
    /** Mean of the Normal distribution */
    real_t mMean;
    /** Standard deviation of the Normal distribution */
    real_t mStdDev;
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
    mMean(mean), mStdDev(stdDev){}

real_t NormalDistribution::mean(){
    return mMean;
}

real_t NormalDistribution::stdDev(){
    return mStdDev;
}

real_t NormalDistribution::pdf(real_t x){
    const real_t log_1_2pi = -1.8378770664093454835606594728112352797227949472;
    return (log_1_2pi - log(mStdDev) - 0.5*pow((x - mMean) / mStdDev, 2) );
}
