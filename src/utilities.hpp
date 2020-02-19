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
    if(a == inf || b == inf)
        return inf;
    if(a == -inf)
        return b;
    if(b == -inf)
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
    cout << "=== " << matrixName << " ===" << endl;
    if(byRow){
        if(cols <= 5){
            for(int x = 0; x < rows; x++){
                cout << "[ ";
                for(int y = 0; y < rows; y++)
                    cout << m[x][y] << " ";
                cout << "]" << endl;
            }
        }
        else{
            cout << "[ ";
            for(int y = 0; y < cols; y++)
                cout << m[0][y] << " ";
            cout << "]" << endl;
            cout << "[ ";
            for(int y = 0; y < cols; y++)
                cout << m[1][y] << " ";
            cout << "]" << endl;
            cout << "..." << endl;
            cout << "[ ";
            for(int y = 0; y < cols; y++)
                cout << m[rows-3][y] << " ";
            cout << "]" << endl;
            cout << "[ ";
            for(int y = 0; y < cols; y++)
                cout << m[rows-2][y] << " ";
            cout << "]" << endl;
            cout << "[ ";
            for(int y = 0; y < cols; y++)
                cout << m[rows-1][y-1] << " ";
            cout << "]" << endl;
        }
    } else {
        if(cols <= 5){
            for(int y = 0; y < cols; y++){
                cout << "[ ";
                for(int x = 0; x < rows; x++)
                    cout << m[x][y] << " ";
                cout << "]" << endl;
            }
        }
        else{
            cout << "[ ";
            for(int x = 0; x < rows; x++)
                cout << m[x][0] << " ";
            cout << "]" << endl;
            cout << "[ ";
            for(int x = 0; x < rows; x++)
                cout << m[x][1] << " ";
            cout << "]" << endl;
            cout << "..." << endl;
            cout << "[ ";
            for(int x = 0; x < rows; x++)
                cout << m[x][cols-3] << " ";
            cout << "]" << endl;
            cout << "[ ";
            for(int x = 0; x < rows; x++)
                cout << m[x][cols-2] << " ";
            cout << "]" << endl;
            cout << "[ ";
            for(int x = 0; x < rows; x++)
                cout << m[x][cols-1] << " ";
            cout << "]" << endl;
        }
    }
}

#endif
