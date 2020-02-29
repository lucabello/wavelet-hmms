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
    cxxopts::Options options(argv[0], "WaHMM - *Wa*velets on *H*idden *M*arkov *M*odels");
    // options
    //   .positional_help("[optional args]")
    //   .show_positional_help();

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
        ("path", "Input file containing the generating path as double separated"
            "by a space",
            cxxopts::value<std::string>(), "<filename>")
        ("binary", "Specify that files for observations and generating path"
            "are in binary format",
            cxxopts::value<bool>())
        ("import", "Input file containing a saved model",
            cxxopts::value<std::string>(), "<filename>")
        ("export", "Export the trained to a file",
            cxxopts::value<std::string>(), "<filename>")
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
        ("v,verbose", "Print extensive algorithms information",
            cxxopts::value<bool>())
        ("h,help", "Print this help message",
            cxxopts::value<bool>())

    //   ("a,apple", "an apple", cxxopts::value<bool>(apple))
    //   ("b,bob", "Bob")
    //   ("t,true", "True", cxxopts::value<bool>()->default_value("true"))
    //   ("f, file", "File", cxxopts::value<std::vector<std::string>>(), "FILE")
    //
    //   ("o,output", "Output file", cxxopts::value<std::string>()
    //       ->default_value("a.out")->implicit_value("b.def"), "BIN")
    //   ("positional",
    //     "Positional arguments: these are the arguments that are entered "
    //     "without an option", cxxopts::value<std::vector<std::string>>())
    //   ("long-description",
    //     "thisisareallylongwordthattakesupthewholelineandcannotbebrokenataspace")
    //   ("help", "Print help")
    //   ("int", "An integer", cxxopts::value<int>(), "N")
    //   ("float", "A floating point number", cxxopts::value<float>())
    //   ("vector", "A list of doubles", cxxopts::value<std::vector<double>>())
    //   ("option_that_is_too_long_for_the_help", "A very long option")
    // #ifdef CXXOPTS_USE_UNICODE
    //   ("unicode", u8"A help option with non-ascii: à. Here the size of the"
    //     " string should be correct")
    // #endif
    ;

    // options.add_options("Group")
    //   ("c,compile", "compile")
    //   ("d,drop", "drop", cxxopts::value<std::vector<std::string>>());

    //options.parse_positional({"input", "output", "positional"});

    auto result = options.parse(argc, argv);

    if (result.count("help")){
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
    //
    // if (apple)
    // {
    //   std::cout << "Saw option ‘a’ " << result.count("a") << " times " <<
    //     std::endl;
    // }
    //
    // if (result.count("b"))
    // {
    //   std::cout << "Saw option ‘b’" << std::endl;
    // }
    //
    // if (result.count("f"))
    // {
    //   auto& ff = result["f"].as<std::vector<std::string>>();
    //   std::cout << "Files" << std::endl;
    //   for (const auto& f : ff)
    //   {
    //     std::cout << f << std::endl;
    //   }
    // }
    //
    // if (result.count("input"))
    // {
    //   std::cout << "Input = " << result["input"].as<std::string>()
    //     << std::endl;
    // }
    //
    // if (result.count("output"))
    // {
    //   std::cout << "Output = " << result["output"].as<std::string>()
    //     << std::endl;
    // }
    //
    // if (result.count("positional"))
    // {
    //   std::cout << "Positional = {";
    //   auto& v = result["positional"].as<std::vector<std::string>>();
    //   for (const auto& s : v) {
    //     std::cout << s << ", ";
    //   }
    //   std::cout << "}" << std::endl;
    // }
    //
    // if (result.count("int"))
    // {
    //   std::cout << "int = " << result["int"].as<int>() << std::endl;
    // }
    //
    // if (result.count("float"))
    // {
    //   std::cout << "float = " << result["float"].as<float>() << std::endl;
    // }
    //
    // if (result.count("vector"))
    // {
    //   std::cout << "vector = ";
    //   const auto values = result["vector"].as<std::vector<double>>();
    //   for (const auto& v : values) {
    //     std::cout << v << ", ";
    //   }
    //   std::cout << std::endl;
    // }
    //
    // std::cout << "Arguments remain = " << argc << std::endl;

    return result;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

// code inside main
// auto result = parse(argc, argv);
// auto arguments = result.arguments();
// std::cout << "Saw " << arguments.size() << " arguments" << std::endl;
//
// return 0;

#endif