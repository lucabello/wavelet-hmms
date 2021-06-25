CC = g++ -std=c++14

HAMMLET_DIR = lib/HaMMLET/src
LOGGER_DIR = lib/aixlog-master/include
LIB_DIR = lib/
INC_FLAGS = -I$(HAMMLET_DIR) -I$(LOGGER_DIR) -I$(LIB_DIR)
ALGORITHMS_DIR = src/algorithms
UTILITIES_DIR = src/utilities
FOLDER_FLAGS = -I$(ALGORITHMS_DIR) -I$(UTILITIES_DIR)

all: ./src/WaHMM.cpp
	$(CC) -o ./bin/WaHMM ./src/WaHMM.cpp $(INC_FLAGS) $(FOLDER_FLAGS) -fpermissive
