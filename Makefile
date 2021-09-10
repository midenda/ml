CC = clang++
CPPFLAGS = -Wall -std=c++17 -g -ggdb
DEBUGFLAGS = -Wall -fsanitize=address -fno-omit-frame-pointer -std=c++17 -g -ggdb
OBJECTS = ml.cpp

all: ml

ml: $(OBJECTS)
	$(CC) $(CPPFLAGS) $(OBJECTS) -o ml

debug: $(OBJECTS)
	$(CC) $(DEBUGFLAGS) $(OBJECTS) -o ml
