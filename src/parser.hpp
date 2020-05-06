/*

Copyright (c) 2014 Jarryd Beck

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/
#ifndef WAHMM_PARSER_HPP
#define WAHMM_PARSER_HPP

#include <iostream>

#include "cxxopts.hpp"
#include "commons.hpp"

cxxopts::ParseResult
parse(int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], "WaHMM - *Wa*velets on *H*idden *M*arkov"
        " *M*odels");

    bool apple = false;

    options.add_options()
        ("s,state", "Specify a state with a Gaussian distribution",
            cxxopts::value<std::vector<double>>(), "<mean>,<stdDev>")
        ("t,transitions", "Transition probabilities in relative terms. For "
            "example a two-state model could have -t 2.0,4.0,8.0,2.0 that "
            "will translate into [0.33, 0.67; 0.8, 0.2]",
            cxxopts::value<std::vector<double>>(), "<a_11>,...")
        ("i,initial", "Initial probability distribution",
            cxxopts::value<std::vector<double>>(), "<pi_1>,...")
        ("obs", "Input file containing the observations as double separated "
            "by a space",
            cxxopts::value<std::string>(), "<filename>")
        ("binary", "Specify that files for observations and generating path"
            "are in binary format",
            cxxopts::value<bool>())
        ("model", "Input file containing a saved model to use for evaluation "
            "and decoding problems",
            cxxopts::value<std::string>(), "<filename>")
        ("estimate", "Input file containing a saved model to use for the "
            "training problem",
            cxxopts::value<std::string>(), "<filename>")
        ("tofile", "Save results as files in the folder ./results/",
            cxxopts::value<bool>())
        ("evaluation", "Solve the evaluation problem using a standard "
            "implementation of the forward algorithm",
            cxxopts::value<bool>())
        ("decoding", "Solve the decoding problem using the Viterbi algorithm",
            cxxopts::value<bool>())
        ("training", "Solve the training problem with the Baum-Welch "
            "algorithm",
            cxxopts::value<bool>())
        ("compressed", "Use the compressed version of the algorithms",
            cxxopts::value<bool>())
        ("silence", "Suppress every message printed to screen",
            cxxopts::value<bool>())
        ("v,verbose", "Print extensive algorithms information",
            cxxopts::value<bool>())
        ("h,help", "Print this help message",
            cxxopts::value<bool>())
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")){
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }

    return result;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

#endif
