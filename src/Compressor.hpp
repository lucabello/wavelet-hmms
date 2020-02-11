#include "Tags.hpp"
#include "HMM.hpp"
//#include "Emissions.hpp"
#include "Blocks.hpp"
//#include "AutoPriors.hpp"
//#include "Records.hpp"
#include "wavelet.hpp"
//#include "StateSequence.hpp"
#include "Statistics.hpp"
#include "includes.hpp"
#include "utils.hpp"

#define INPUT_FILENAME std::string("data")

class Compressor {
    /** Input observations values */
    vector<real_t> mInputValues;
    /** Sum of the observations in a block */
    vector<SufficientStatistics<Normal>> mStats;
    /** Threshold used to form blocks in the breakpoint array */
    real_t mThreshold;
    /** Integral array, @see HaMMLET documentation */
    Statistics<IntegralArray, Normal> *mIntegralArray;
    /** Breakpoint array, @see HaMMLET documentation */
    Blocks<BreakpointArray> *mWaveletBlocks;
public:
    Compressor(const Compressor& that) = delete;
    /**
    * Construct a Compressor object. Data is read from an input file, creating
    * a breakpoint array and defining a threshold to use; blocks are then
    * defined and the integral array is created. There is initialization to
    * correctly read the first block.
    *
    * @param filename the name of the input file to read the data
    */
    Compressor(std::string& filename);
    ~Compressor();
    /** Move the "current block pointer" back to the first block. */
    void initForward();
    /** Move the "current block pointer" one step forward. */
    void next();
    /** Return the start index of the "current block". */
    size_t start();
    /** Return the end index of the "current block". */
    size_t end();
    /** Return the size of the "current block". */
    size_t blockSize();
    /** Return the sum of the observations in the "current block". */
    real_t blockSum();
    /** Print start and end indexes of the "current block" alongside with its
    * size and the sum of the observation values in it; the format used is
    * [start,end) size - "Sum:" sum
    */
    void printBlockInfo();
    /**
    * Print block information for all blocks, preceded by the thresold value
    * used to form them.
    */
    void printAllBlocks();
};

Compressor::Compressor(std::string& f){
    try {
        const size_t nrDataDim = 1; // number of dimensions
        ifstream finput(f);
        // Open the file and populate mInputValues
        if(finput) {
            MaxletTransform(finput, mInputValues, mStats, nrDataDim,
                mInputValues.size() + nrLinesInFile(finput) + 1);
        } else {
          throw runtime_error( "Cannot read from input file " + f + "!" );
        }

        // compute an estimate of the noise variance from the finest
        // detail coefficients
		double stdEstimate = 0;
		size_t nrDetailCoeffs = 0;
		for (size_t i = 1; i < mInputValues.size(); i += 2){
			stdEstimate += mInputValues[i];
			nrDetailCoeffs++;
		}
		stdEstimate /= nrDetailCoeffs;	// yields mean absolute deviation
        // divide by sqrt(2/pi) to get estimate of standard deviation
        // for normal distribution
		stdEstimate /= 0.797884560802865355879892119868763736951717262329869315331;
        mThreshold = sqrt(2 * log((real_t)mWaveletBlocks->size()) * stdEstimate);

        HaarBreakpointWeights(mInputValues);
		mIntegralArray = new Statistics<IntegralArray, Normal>(mStats, nrDataDim);
		mWaveletBlocks = new Blocks<BreakpointArray>(mInputValues);
		mWaveletBlocks->createBlocks(mThreshold);
		mWaveletBlocks->initForward();
		mWaveletBlocks->next();
    }
    catch(exception& e) {
        std::cout << std::flush;
		cerr << endl << flush << "[ERROR] " << e.what()  << endl << flush;
        throw e;
    }
}

Compressor::~Compressor(){
    delete mIntegralArray;
    delete mWaveletBlocks;
}

void Compressor::initForward(){
    mWaveletBlocks->initForward();
    mWaveletBlocks->next();
}

void Compressor::next(){
    mWaveletBlocks->next();
}

size_t Compressor::start(){
    mWaveletBlocks->start();
}

size_t Compressor::end(){
    mWaveletBlocks->end();
}

size_t Compressor::blockSize(){
    mWaveletBlocks->blockSize();
}

real_t Compressor::blockSum(){
    mIntegralArray->setStats(*mWaveletBlocks);
    return mIntegralArray->suffStat(0).sum(); // 0 is the dimension index
}

void Compressor::printBlockInfo(){
    mWaveletBlocks->printBlock();
    cout << "- Sum: " << blockSum() << endl;
}

void Compressor::printAllBlocks(){
    cout << "Threshold used: " << mThreshold << endl;
    do {
        printBlockInfo();
        mWaveletBlocks->next();
    } while (mWaveletBlocks->end() < mWaveletBlocks->size());
    initForward();
}
