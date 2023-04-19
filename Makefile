# Makefile for COMP4300/8300 Assignment 1
# Peter Strazdins, SOCO ANU, Feb 21
.SUFFIXES:
.PRECIOUS: %.o

HDRS=serAdvect.h parAdvect.h
OBJS=serAdvect.o parAdvect.o
PROG=testAdvect
FFT_CONV_KERNEL=0
CCFLAGS=-O3 -DFFT_CONV_KERNEL=$(FFT_CONV_KERNEL)
LINKERFLAGS=

ifeq ($(FFT_CONV_KERNEL), 1)
	LINKERFLAGS=$(shell pkg-config --cflags --libs fftw3)
endif

all: $(PROG) 

%: %.o $(OBJS)
	mpicc -o $* $*.o $(OBJS) $(LINKERFLAGS) -lm
%.o: %.c $(HDRS)
	mpicc -Wall $(CCFLAGS) -c $*.c
startup:
	mpicc -o startup startup.o -lm
clean:
	rm -f *.o $(PROG)
