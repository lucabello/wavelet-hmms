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
    //TODO: write more efficient implementation
    const real_t log_sqrt2pi = 0.9189385332046727417803297364056176398613974;
    return (- log_sqrt2pi - log(mStdDev) - 0.5*pow((x - mMean) / mStdDev, 2) );
    //const real_t sqrt_2pi = 2.50662827463100050241576528481104525300698674060;
    //return log(exp(-(pow(x - mMean, 2))/
    //    (2*pow(mStdDev, 2)))/(mStdDev*sqrt_2pi));
}
