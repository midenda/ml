CC = clang++
CPPFLAGS = -Wall -std=c++17
OBJECTS = ml.cpp

all: ml

ml: $(OBJECTS)
	$(CC) $(CPPFLAGS) $(OBJECTS) -o ml
