# Includes for gflags only for the millenium mayhem
ifneq ($(wildcard /home/eecs/agnusmaximus/gflags/build/include/.*),)
	LIBS += -I/home/eecs/agnusmaximus/gflags/build/include
endif
ifneq ($(wildcard /home/eecs/agnusmaximus/gflags/build/lib/.*),)
	LIBS += -L/home/eecs/agnusmaximus/gflags/build/lib
endif

# Require libraries
LIBS += -lpthread -lgflags -fopenmp

# Flags
FLAGS=-Ofast -std=c++11

# Select compiler (mac uses clang-omp++). Default is g++.
CC=g++
CLANG_OMP++ := $(shell command clang-omp++ --version 2> /dev/null)
ifdef CLANG_OMP++
    CC=clang-omp++
endif

FILES=src/main.cpp

all:
	$(CC) $(FLAGS) $(FILES) $(LIBS) -o cyclades
