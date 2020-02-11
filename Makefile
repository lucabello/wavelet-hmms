CC = g++

HMM_DIR = ../../Downloads/HaMMLET/src
HFLAGS=-I$(HMM_DIR)

all:
	$(CC) -o ./bin/WaHMM ./src/WaHMM.cpp $(HFLAGS)
