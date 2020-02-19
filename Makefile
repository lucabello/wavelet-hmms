CC = g++ -std=c++14

HAMMLET_DIR = lib/HaMMLET/src

wahmm: ./src/WaHMM.cpp
	$(CC) -o ./bin/WaHMM ./src/WaHMM.cpp -I$(HAMMLET_DIR)

parser:
	$(CC) -o ./bin/parserTest ./src/main.cpp -Ilib/ -fpermissive
