#ifndef WAHMM_COMMONS_HPP
#define WAHMM_COMMONS_HPP

#define PATH_OUT std::string("results/")
#define MIN_VARIANCE 0.01

#include <cstdint>

#include <cstddef>
using std::size_t;

#include <vector>
using std::vector;

#include <map>

#include <string>
using std::string;
using std::to_string;

#include <iostream>
using std::istream;
using std::ostream;
using std::endl;
using std::cin;
using std::cout;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <cmath>
using std::pow;
using std::exp;	// e^x
using std::log;	// natural log
using std::sqrt;
using std::abs;

#include <math.h>

using std::isfinite;

#include <algorithm>
using std::min;
using std::max;

namespace wahmm {
    typedef double real_t;
}
const wahmm::real_t infin = std::numeric_limits<wahmm::real_t>::infinity();

struct blockdata_t {
    size_t nw;
    wahmm::real_t s1;
    wahmm::real_t s2;
};
typedef blockdata_t blockdata;

#endif
