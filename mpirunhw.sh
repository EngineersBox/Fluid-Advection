#!/bin/sh

mpirun --hostfile hostfile --use-hwthread-cpus $@
