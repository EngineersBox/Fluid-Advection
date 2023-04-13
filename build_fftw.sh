#!/bin/sh

make clean
make LINKERFLAGS="$(pkg-config --cflags --libs fftw3)" FFT_CONV_KERNEL=1 all
