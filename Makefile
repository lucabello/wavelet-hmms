CC = g++ -std=c++14

HAMMLET_DIR = lib/HaMMLET/src
LIB_DIR = lib/
INC_FLAGS = -I$(HAMMLET_DIR) -I$(LIB_DIR)

all: ./src/WaHMM.cpp
	$(CC) -o ./bin/WaHMM ./src/WaHMM.cpp $(INC_FLAGS) -fpermissive
