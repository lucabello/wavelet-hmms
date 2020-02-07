CC = g++
INC_DIR = ../../Downloads/HaMMLET/src
CFLAGS=-I$(INC_DIR)

all:
	$(CC) -o ./bin/main ./src/main.cpp $(CFLAGS)
