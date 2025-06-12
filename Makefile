CC = clang++
CPPFLAGS = -Wall -std=c++2c -g -ggdb
DEBUGFLAGS = -Wall -fsanitize=address -fno-omit-frame-pointer -std=c++2c -g -ggdb
HEADERS = ml.h tensor.h benchmark.h regression.h
OBJECTS = train.cpp
TEST_HEADERS = tests/
TESTS = tests.cpp

all: test

ml: $(OBJECTS) $(HEADERS)
	$(CC) $(CPPFLAGS) $(OBJECTS) -o ml

debug: $(OBJECTS) $(HEADERS)
	$(CC) $(DEBUGFLAGS) $(OBJECTS) -o debug

test: $(TESTS) $(HEADERS) $(TEST_HEADERS)
	$(CC) $(CPPFLAGS) $(TESTS) -o test
