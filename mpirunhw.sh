#!/bin/sh

mpirun --hostfile hostfile --use-hwthread-cpus --tag-output $@
