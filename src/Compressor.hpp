// #include "hammlet-manpage.hpp"
#include "Tags.hpp"
#include "HMM.hpp"
// #include "Options.hpp"
#include "Parser.hpp"
#include "Emissions.hpp"
#include "Blocks.hpp"
#include "AutoPriors.hpp"
#include "Records.hpp"
#include "wavelet.hpp"
#include "StateSequence.hpp"
#include "Statistics.hpp"
#include "includes.hpp"
#include "utils.hpp"

#define INPUT_FILENAME std::string("data")

class Compressor {
    vector<real_t> inputValues;
    vector<SufficientStatistics<Normal>> stats;
    real_t threshold;
    Statistics<IntegralArray, Normal> *integralArray;
    Blocks<BreakpointArray> *waveletBlocks;
public:
    Compressor(const Compressor& that) = delete;
    Compressor(std::string& filename);
    ~Compressor();
    void initForward();
    void next();
    void start();
    void end();
    void blockSize();
    real_t blockSum();
    void printBlockInfo();
    void printAllBlocks();
};

Compressor::Compressor(std::string& f){
    try {
        const size_t nrDataDim = 1;
        ifstream finput(f);
        // Open the file and populate inputValues
        if(finput) {
            MaxletTransform(finput, inputValues, stats, nrDataDim,
                inputValues.size() + nrLinesInFile(finput) + 1);
        } else {
          throw runtime_error( "Cannot read from input file " + f + "!" );
        }

        // compute an estimate of the noise variance from the finest
        // detail coefficients
		double stdEstimate = 0;
		size_t nrDetailCoeffs = 0;
		for (size_t i = 1; i < inputValues.size(); i += 2){
			stdEstimate += inputValues[i];
			nrDetailCoeffs++;
		}
		stdEstimate /= nrDetailCoeffs;	// yields mean absolute deviation
        // divide by sqrt(2/pi) to get estimate of standard deviation
        // for normal distribution
		stdEstimate /= 0.797884560802865355879892119868763736951717262329869315331;
        threshold = sqrt(2 * log((real_t)waveletBlocks->size()) * stdEstimate);

        HaarBreakpointWeights(inputValues);
		integralArray = new Statistics<IntegralArray, Normal>(stats, nrDataDim);
		waveletBlocks = new Blocks<BreakpointArray>(inputValues);
		waveletBlocks->createBlocks(threshold);
		waveletBlocks->initForward();
		waveletBlocks->next();
    }
    catch(exception& e) {
        std::cout << std::flush;
		cerr << endl << flush << "[ERROR] " << e.what()  << endl << flush;
        throw e;
    }
}

Compressor::~Compressor(){
    delete integralArray;
    delete waveletBlocks;
}

void Compressor::initForward(){
    waveletBlocks->initForward();
    waveletBlocks->next();
}

void Compressor::next(){
    waveletBlocks->next();
}

void Compressor::start(){
    waveletBlocks->start();
}

void Compressor::end(){
    waveletBlocks->end();
}

void Compressor::blockSize(){
    waveletBlocks->blockSize();
}

real_t Compressor::blockSum(){
    integralArray->setStats(*waveletBlocks);
    return integralArray->suffStat(0).sum(); // 0 is the dimension index
}

void Compressor::printBlockInfo(){
    waveletBlocks->printBlock();
    cout << "- Sum: " << blockSum() << endl;
}

void Compressor::printAllBlocks(){
    cout << "Threshold used: " << threshold << endl;
    cout << "Printing all blocks with the following format: " << endl;
    cout << "[start,end) size" << endl;
    do {
        printBlockInfo();
        waveletBlocks->next();
    } while (waveletBlocks->end() < waveletBlocks->size());
    initForward();
}

// 		cout << "---- Blocks -----" << endl;
// 		for(int i = 0; waveletBlocks.end() < waveletBlocks.size() ; i++){
// 			waveletBlocks.printBlock();
// 			ia.setStats(waveletBlocks);
// 			cout << endl;
// 			// 1 is the number of dimensions
// 			cout << "suffStat[0]: " << ia.suffStat(0) << endl;
// 			waveletBlocks.next();
// 		}
//
// 		cin >> a;
//
//     }
// }
//
// int main( int argc, const char* argv[] ) {
// 	int a;
// 	try {
//
// 		vector<real_t> inputValues;
// 		vector<SufficientStatistics<Normal>> stats;
// 		ifstream fin( INPUT_FILENAME );
// 		bool verbose = true;
// 		const size_t nrDataDim = 1;
//
// 		if ( fin ) {
// 			// TODO this can still lead to reallocation, fix later
// 			// TODO MaxletTransform does not work for multiple files in its current state
// 			MaxletTransform( fin, inputValues, stats, nrDataDim, inputValues.size() + nrLinesInFile( fin ) + 1 );
// 			// NOTE Reserving +1 is really important here! In the integral array, an element is appended to stats, and not reserving space for that element can lead to reallocations in the gigabyte range!
// 		} else {
// 			throw runtime_error( "Cannot read from input file " + INPUT_FILENAME + "!" );
// 		}
// 		const size_t T = inputValues.size();
//
// 		cout << "Number of data points: " + to_string( T ) << endl << flush;
//
// 		// compute an estimate of the noise variance from the finest detail coefficients
// 		double stdEstimate=0;
// 		size_t nrDetailCoeffs=0;
// 		for (size_t i=1; i< inputValues.size(); i+=2){
// 			stdEstimate += inputValues[i];
// 			nrDetailCoeffs++;
// 		}
// 		stdEstimate /= nrDetailCoeffs;	// yields mean absolute deviation
// 		stdEstimate /= 0.797884560802865355879892119868763736951717262329869315331; // divide by sqrt(2/pi) to get estimate of standard deviation for normal distribution
//
// 		cout << "stdEstimate: " + to_string( stdEstimate ) << endl << flush;
// 		// inputValues holds the maxlet transform, now transform it to breakpoint weights
// 		if ( verbose ) {
// 			cout << "Calculating Haar breakpoint weights" << endl << flush;
// 		}
// 		HaarBreakpointWeights( inputValues );
//
// 		typedef Statistics<IntegralArray, Normal> S;
// 		typedef Blocks<BreakpointArray> B;
// 		S ia( stats, nrDataDim );
// 		B waveletBlocks( inputValues );
//
// 		cout << "---- Trying to create blocks ----" << endl;
// 		cout << "Threshold: " << sqrt( 2 * log( ( real_t )waveletBlocks.size() ) * stdEstimate) << endl;
// 		waveletBlocks.createBlocks(sqrt( 2 * log( ( real_t )waveletBlocks.size() ) * stdEstimate));
// 		waveletBlocks.initForward();
// 		waveletBlocks.next();
//
//
// 		cout << "---- BreakpointArray ----" << endl;
// 		cout << "avgWeight: " + to_string ( waveletBlocks.avgWeight() ) << endl;
// 		cout << "start: " + to_string( waveletBlocks.start() ) << endl;
// 		cout << "end: " + to_string( waveletBlocks.end() ) << endl;
// 	  cout << "pos: " + to_string( waveletBlocks.pos() ) << endl;
// 		cout << "blockSize: " + to_string( waveletBlocks.blockSize() ) << endl;
// 		waveletBlocks.printBlock();
// 		cout << endl;
//
// 		cout << "---- Blocks -----" << endl;
// 		for(int i = 0; waveletBlocks.end() < waveletBlocks.size() ; i++){
// 			waveletBlocks.printBlock();
// 			ia.setStats(waveletBlocks);
// 			cout << endl;
// 			// 1 is the number of dimensions
// 			cout << "suffStat[0]: " << ia.suffStat(0) << endl;
// 			waveletBlocks.next();
// 		}
//
// 		cin >> a;
//
// 	} catch
// 		( exception& e ) {
// 		cout << flush;
// 		cerr << endl << flush << "[ERROR] " << e.what()  << endl;
// 		cerr << "Terminating HaMMLET. The rest is silence." << endl << flush;
//
// 		return 1;
// 	}
//
// 	return 0;
//
// }
