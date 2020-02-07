#include "Compressor.hpp"

int main(int argc, const char* argv[]){
    std::string filename("data");
    Compressor comp(filename);
    comp.printAllBlocks();

    return 0;
}
