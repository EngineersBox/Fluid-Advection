# Fluid Advection MPI

Implementation of fluid advection solver using MPI for massively parallel distributed architectures.


## Setup

The project contains a test program `testAdvect.c`, a file `serAdvect.c` containing a serial advection solver and support functions, some header files, and a template parallel advection solver `parAdvect.c`. The test program can be built using the command `make`. 

It also contains a report writeup `ps-ass1Rep.pdf`, detailing discussion of questions presented in the assignment description in [docs/assignmet_outline.md](./docs/assignment_outline.md). 

The usage for the test program is: 

```bash
mpirun -np p ./testAdvect [-P P] [-w w] [-o] [-x] M N [r] 
```

with default values `P=p, w=1, r=1`. This will run an `M` by `N` advection simulation over `r` timesteps (repetitions) using a `P` by `Q` process grid, where `p=PQ`. `w` specifies the halo width (normally it is 1). If `-o` is specified, halo communication should be overlapped with computation. The `-x` is used to invoke an optional extra optimization. 

There is also a `-v` option which can be used for debugging (try using `-v 1`, `-v 2` etc). 

The test program initializes the local field array `u` with leading dimension `ldu`, calls the appropriate parallel advection function (in `parAdvect.c`), and determines the error in the final field. It assumes a 2D block distribution of the global advection field. However, `parAdvect.c` determines the details of this distribution (i.e. how to deal with remainders), and exports: 

* `M_loc`, `N_loc`: the local advection field size (excluding the halo). 
* `M0`, `N0`: the local field element `(0,0)` is global element `(M0,N0)` 
