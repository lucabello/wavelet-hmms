#ifndef WAHMM_UTILITIES_HPP
#define WAHMM_UTILITIES_HPP
#include <iostream>
#include "commons.hpp"

/** Given (a = log(x) and b = log(y), returns log(x+y)) */
wahmm::real_t sum_logarithms(wahmm::real_t a, wahmm::real_t b){
    /**
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	*/
    if(a == infin || b == infin)
        return infin;
    if(a == -infin)
        return b;
    if(b == -infin)
        return a;
    if(a > b)
        return a + log1pf(exp(b-a));
    return b + log1pf(exp(a-b));
}

/** Free a matrix of wahmm::real_t with 'rows' rows. */
void freeMatrix(wahmm::real_t** m, size_t rows){
    for(int i = 0; i < rows; i++)
        delete[] m[i];
    delete[] m;
}

/** Compute K(n_w,j) */
wahmm::real_t compute_k(Model& m, size_t j, size_t nw){
    const wahmm::real_t log_2pi = 1.8378770664093454835606594728112352797227949;
    const wahmm::real_t log_2pi_over2 = log_2pi/2;

    if(m.mKValues.count(nw) != 0){
        return m.mKValues.find(nw)->second.at(j);
    }

    // compute K(n_w, j) for all states and save the value in the map
    m.mKValues[nw] = std::vector<wahmm::real_t>();
    for(size_t i = 0; i < m.mStates.size(); i++){
        wahmm::real_t first = (nw-1)*m.mLogTransitions[i][i];
        wahmm::real_t second = nw*(m.mStates[i].logStdDev() +
            pow(m.mStates[i].mean()/m.mStates[i].stdDev(),2)/2 +
            log_2pi_over2);
        m.mKValues[nw].push_back(first - second);
    }

    return m.mKValues.find(nw)->second.at(j);

}

/** Compute E_w(j) */
wahmm::real_t compute_e(Model& m, size_t j, blockdata bd){
    wahmm::real_t num = 2*m.mStates[j].mean()*bd.s1 - bd.s2;
    wahmm::real_t den = 2*pow(m.mStates[j].stdDev(),2);
    return (num/den) + compute_k(m, j, bd.nw);
}

/**
* Prints a matrix in a compressed form with the following format (example 10x3):
* [ x_00 x_01 x_02 ]
* [ x_10 x_11 x_12 ]
* ...
* [ x_70 x_71 x_72 ]
* [ x_80 x_81 x_82 ]
* [ x_90 x_91 x_92 ]
* If the matrix is smaller than 5 rows, it will be printed fully.
* @param m the matrix to print
* @param rows
* @param cols
* @param matrixName printed before the matrix as a header
* @param byRow if true prints by row, otherwise by columns
*/
void printMatrixSummary(wahmm::real_t **m, size_t rows, size_t cols,
    std::string matrixName, bool byRow){
    cout << "[>] === " << matrixName << " ===" << endl;
    if(byRow){
        if(cols <= 5){
            for(int x = 0; x < rows; x++){
                for(int y = 0; y < rows; y++)
                    cout << m[x][y] << "\t";
                cout << endl;
            }
        }
        else{
            for(int y = 0; y < cols; y++)
                cout << m[0][y] << "\t";
            cout << endl;
            for(int y = 0; y < cols; y++)
                cout << m[1][y] << "\t";
            cout << endl;
            cout << "..." << endl;
            for(int y = 0; y < cols; y++)
                cout << m[rows-3][y] << "\t";
            cout << endl;
            for(int y = 0; y < cols; y++)
                cout << m[rows-2][y] << "\t";
            cout << endl;
            for(int y = 0; y < cols; y++)
                cout << m[rows-1][y-1] << "\t";
            cout << endl;
        }
    } else {
        if(cols <= 5){
            for(int y = 0; y < cols; y++){
                for(int x = 0; x < rows; x++)
                    cout << m[x][y] << "\t";
                cout << endl;
            }
        }
        else{
            for(int x = 0; x < rows; x++)
                cout << m[x][0] << "\t";
            cout << endl;
            for(int x = 0; x < rows; x++)
                cout << m[x][1] << "\t";
            cout << endl;
            cout << "..." << endl;
            for(int x = 0; x < rows; x++)
                cout << m[x][cols-3] << "\t";
            cout << endl;
            for(int x = 0; x < rows; x++)
                cout << m[x][cols-2] << "\t";
            cout << endl;
            for(int x = 0; x < rows; x++)
                cout << m[x][cols-1] << "\t";
            cout << endl;
        }
    }
    cout << "======" << endl;
}

#endif
