#ifndef WAHMM_COMMONS_HPP
#define WAHMM_COMMONS_HPP

#include <cstdint>

#include <cstddef>
using std::size_t;

#include <vector>
using std::vector;

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

// #include <stdexcept>
// using std::runtime_error;	// TODO throw the appropriate errors, like logic_error etc.
// using std::exception;

#include <cmath>
using std::pow;
using std::exp;	// e^x
using std::log;	// natural log
using std::sqrt;
using std::abs;

using std::isfinite;

#include <algorithm>
using std::min;
using std::max;

namespace wahmm {
    typedef double real_t;
}
const wahmm::real_t infin = std::numeric_limits<wahmm::real_t>::infinity();

#endif
