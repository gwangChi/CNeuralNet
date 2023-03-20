CC = g++
CFLAGS = -std=c++20

serial_neuro: neuro.o
	$(CC) $(CFLAGS) -o serial_neuro neuro.o

neuro.o: neuro.cpp mnist.h
	$(CC) $(CFLAGS) -c neuro.cpp