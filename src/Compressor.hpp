#ifndef WAHMM_COMPRESSOR_HPP
#define WAHMM_COMPRESSOR_HPP

#include "includes.hpp"
#include "Tags.hpp"
#include "HMM.hpp"
#include "Blocks.hpp"
#include "wavelet.hpp"
#include "Statistics.hpp"
#include "utils.hpp"
#include "commons.hpp"

template< typename T>
void MaxletTransform(FILE* fin,
    bool binary,
    vector<real_t>& coeffs,
    vector< SufficientStatistics<T> >& suffstats,
    const size_t nrDim = 1,
    const size_t reserveT = 0 // an estimate of the number of data points to avoid reallocation
);

/**
* Class constituting the interface between WaHMM and HaMMLET. It serves the
* purpose of reading the observations and compressing them into blocks,
* also providing an interace to navigate easily the data structures that
* HaMMLET offers.
*/
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
    size_t mBlocksNumber;
    /** Stack to hold the reverse order of the blocks, built on command */
    std::list<blockdata> mReverseList;
    /** Iterator to navigate the reverse list */
    std::list<blockdata>::iterator listIt = mReverseList.begin();
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
    Compressor(std::string& filename, bool binary);
    ~Compressor();
    /** Move the "current block pointer" back to the first block. */
    void initForward();
    /**
    * Move the "current block pointer" one step forward.
    * @return false if the "current block" is the last one
    */
    bool next();
    /** Return the start index of the "current block". */
    size_t start();
    /** Return the end index of the "current block". */
    size_t end();
    /** Return the size of the "current block". */
    size_t blockSize();
    /** Return the sum of the observations in the "current block". */
    wahmm::real_t blockSum();
    /** Return the squared sum of the observations in the "current block". */
    wahmm::real_t blockSumSq();
    /** Return the data associated with the current block. */
    blockdata blockData();
    /** Initialize the iterator for the reverse list. */
    void initBackward();
    /**
    * Advance the reverse list iterator.
    * @return false if the iterator reaches the end of the list
    */
    bool reverseNext();
    /** Get the block size from the reverse list iterator. */
    size_t reverseSize();
    /** Return the sum of the observations using the reverse block iterator. */
    wahmm::real_t reverseSum();
    /** Return the squared sum of observations in the reverse iterator. */
    wahmm::real_t reverseSumSq();
    /** Return the data associated with the block in reverse iterator. */
    blockdata reverseBlockData();
    /** Return the number of blocks. */
    size_t blocksNumber();
    /** Return the number of observations. */
    size_t observationsNumber();
    /** Build a list with the blocks in reverse order */
    void buildReverse();
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

Compressor::Compressor(std::string& f, bool binary){
    try {
        const size_t nrDataDim = 1; // number of dimensions
        FILE* fin;
        if(!binary)
            fin = fopen(f.c_str(), "r");
        else
            fin = fopen(f.c_str(), "rb");
        // Open the file and populate mInputValues
        if(fin != NULL) {
            MaxletTransform(fin, binary, mInputValues, mStats, nrDataDim,
                mInputValues.size() + 2);
        } else {
          throw runtime_error( "Cannot read from input file " + f + "!" );
        }
        fclose(fin);

        // compute an estimate of the noise variance from the finest
        // detail coefficients
		double stdEstimate = 0;
        double estimateAccum = 0;
		for (size_t i = 1; i < mInputValues.size(); i += 2){
			stdEstimate += mInputValues[i];
            if(stdEstimate > mInputValues.size()){
                estimateAccum += stdEstimate/(mInputValues.size()/2);
                stdEstimate = 0;
            }
		}
        estimateAccum += stdEstimate/(mInputValues.size()/2);
        stdEstimate = estimateAccum;

        HaarBreakpointWeights(mInputValues);
        mIntegralArray = new Statistics<IntegralArray, Normal>(mStats, nrDataDim);
        mWaveletBlocks = new Blocks<BreakpointArray>(mInputValues);

        // std::cout << "stdEstimate: " << stdEstimate << std::endl;
        //stdEstimate /= nrDetailCoeffs;	// yields mean absolute deviation
        //std::cout << "stdEstimate: " << stdEstimate << std::endl;
        // divide by sqrt(2/pi) to get estimate of standard deviation
        // for normal distribution
		stdEstimate /= 0.797884560802865355879892119868763736951717262329869315331;
        // std::cout << "stdEstimate: " << stdEstimate << std::endl;
        // std::cout << "log(mWaveletBlocks.size()): " << log(mWaveletBlocks->size()) << std::endl;
        // std::cout << "stdEstimate: " << stdEstimate << std::endl;
        mThreshold = sqrt(2 * log((real_t)mWaveletBlocks->size()) * stdEstimate);

        //std::cout << "Using this threshold: " << mThreshold << std::endl;
		mWaveletBlocks->createBlocks(mThreshold); //mThreshold
		mWaveletBlocks->initForward();
		mWaveletBlocks->next();
        mBlocksNumber = 0;
        do {
            mBlocksNumber++;
        } while(mWaveletBlocks->next());
        // std::cout << "number of blocks: " << mBlocksNumber << std::endl;
        mWaveletBlocks->initForward();
        mWaveletBlocks->next();
    }
    catch(exception& e) {
        std::cout << std::flush;
		cerr << endl << flush << "[CompressorError] " << e.what()  << endl << flush;
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

bool Compressor::next(){
    return mWaveletBlocks->next();
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

wahmm::real_t Compressor::blockSum(){
    mIntegralArray->setStats(*mWaveletBlocks);
    return mIntegralArray->suffStat(0).sum(); // 0 is the dimension index
}

wahmm::real_t Compressor::blockSumSq(){
    mIntegralArray->setStats(*mWaveletBlocks);
    return mIntegralArray->suffStat(0).sumSq(); // 0 is the dimension index
}

blockdata Compressor::blockData(){
    blockdata bd = {blockSize(), blockSum(), blockSumSq()};
    return bd;
}

void Compressor::initBackward(){
    listIt = mReverseList.begin();
}

bool Compressor::reverseNext(){
    if(listIt == mReverseList.end())
        return false;
    listIt++;
    return true;
}

size_t Compressor::reverseSize(){
    return (*listIt).nw;
}

wahmm::real_t Compressor::reverseSum(){
    return (*listIt).s1;
}

wahmm::real_t Compressor::reverseSumSq(){
    return (*listIt).s2;
}

blockdata Compressor::reverseBlockData(){
    blockdata bd = {(*listIt).nw, (*listIt).s1, (*listIt).s2};
    return bd;
}

size_t Compressor::blocksNumber(){
    return mBlocksNumber;
}

size_t Compressor::observationsNumber(){
    return mInputValues.size();
}

void Compressor::buildReverse(){
    mWaveletBlocks->initForward();
    blockdata bd;
    while(mWaveletBlocks->next()){
        bd.nw = blockSize();
        bd.s1 = blockSum();
        bd.s2 = blockSumSq();
        mReverseList.push_front(bd);
    }
    mWaveletBlocks->initForward();
}

void Compressor::printBlockInfo(){
    mWaveletBlocks->printBlock();
    cout << "- Sum: " << blockSum();
    cout << "- SumSq: " << blockSumSq() << endl;
}

void Compressor::printAllBlocks(){
    do {
        printBlockInfo();
        mWaveletBlocks->next();
    } while (mWaveletBlocks->end() < mWaveletBlocks->size());
    printBlockInfo();
    cout << "Threshold used: " << mThreshold << endl;
    initForward();
}

/*
* Overloads the original function in HaMMLET to use C-style file input
* for efficiency reasons
*/
template< typename T>
void MaxletTransform(
    FILE* fin,
    bool binary,
    vector<real_t>& coeffs,
    vector< SufficientStatistics<T> >& suffstats,
    const size_t nrDim = 1,
    const size_t reserveT = 0	// an estimate of the number of data points to avoid reallocation
) {

	if ( nrDim <= 0 ) {
		throw runtime_error( "Number of dimensions must be positive!" );
	}


	if ( coeffs.size() > 0 ) {
		throw runtime_error( "Coefficient array must be empty!" );
	}

	if ( suffstats.size() > 0 ) {
		throw runtime_error( "Statistics array must be empty!" );
	}

	if ( fin != NULL ) {


		coeffs.reserve( ( reserveT + nrDim ) / nrDim + nrDim );
		suffstats.reserve( reserveT + nrDim );

// 	stack<real_t, vector<real_t> > S;	// stack never gets larger than nrDim*log2(T), so we don't expect a lot of reallocation, and save a lot of push and pop operations due to random access
		vector<real_t> S;
		size_t i = 0;
		real_t v = 0;
		size_t dim = 0;
        double inputNum = 0;

        bool fileEnd = false;
        if(!binary){
            fileEnd = (fscanf(fin, "%lf", &inputNum) == EOF);
        }
        else {
            fileEnd = (fread(&inputNum,1,sizeof(double),fin) != sizeof(double));
        }
		while ( !fileEnd ) {
            v = (real_t)inputNum;
			S.push_back( v );
			suffstats.push_back( SufficientStatistics<T>( v ) );
			dim++;	// set dimension of next value
			if ( dim == nrDim ) {	// filled all dimensions at index i
				dim = 0;	// next value will be first dimension again


				coeffs.push_back( inf );


				size_t j = i;	// points to node indices on an upward-left path (i.e. DFS post-order)
				size_t m = 1;	// mask to determine whether j is an index of a left child
				real_t normalizer = sqrt2half;

				while ( ( j & m ) > 0 ) {	// while j is on a left-upward path (DFS post-order)

					real_t maxCoeff = 0;	// the maximum detail coefficient across dimensions at j; NOTE we cannot take the maximum with coeffs because it contains infinity

					size_t L = S.size() - 2 * nrDim;	// index of left element in stack, get incremented to iterate over dimensions
					size_t R = L + nrDim;	// likewise, index of right element in stack


					// compute maximum of detail coefficients across dimensions
					for ( size_t d = 0; d < nrDim; ++d ) {
						maxCoeff = max( maxCoeff, normalizer * abs( S[L] - S[R] ) );
						S[L] += S[R];	// add right values to left values, so only the right values need to be popped
						L++;		// go to next dimension
						R++;
					}
					coeffs[j] = maxCoeff;


					// pop the right values
					for ( size_t d = 0; d < nrDim; ++d ) {
						S.pop_back();
					}


					j = j - m;	// move to left parent (if current position is not a right child, the loop will exit)
					m *= 2;	// move bit-mask to the left, i.e. check if i is still on a left-up path)
					normalizer *= sqrt2half;	// moving up one level changes normalization factor
				}
				i++;
			}
            if(!binary){
                fileEnd = (fscanf(fin, "%lf", &inputNum) == EOF);
            }
            else {
                fileEnd = (fread(&inputNum,1,sizeof(double),fin) != sizeof(double));
            }
		}


		if ( dim != 0 ) {
			throw runtime_error( "Input stream did not contain enough values to fill all dimensions at last position!" );
		}

		coeffs[0] = inf;

	} else {
		throw runtime_error( "Cannot read input file or stream!" );
	}
}


#endif
