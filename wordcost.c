/* ping_pong.c -- two-process ping-pong -- send from 0 to 1
 * and send back from 1 to 0
 * See Chap 12, pp. 267 & ff. in PPMPI */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#define BUFFER_SIZE 100
#define MAX 4
#define TS_OVERHEAD 0.0002951666666667

int main(int argc, char** argv) {
	int p,my_rank;
	double x[BUFFER_SIZE];
	double start, finish;
	double raw_time;
	MPI_Status status;
	MPI_Comm comm;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_dup(MPI_COMM_WORLD, &comm);
	srand(time(NULL));
	for (size_t i = 0; i < BUFFER_SIZE; i++) {
		x[i] = (double) rand() / (double) rand();
	}

	double wtime_overhead = 0.0;
	for (size_t i = 0; i < 100; i++) {
		start = MPI_Wtime();
		finish = MPI_Wtime();
		wtime_overhead += start - finish;
	}
	wtime_overhead /= 100.0;
	if (my_rank == 0) {
		for (size_t pass = 0; pass < MAX; pass++) {
			MPI_Barrier(comm);
			start = MPI_Wtime();
			MPI_Send(x, BUFFER_SIZE, MPI_DOUBLE,1,0,comm);
			finish = MPI_Wtime();
			MPI_Recv(x, BUFFER_SIZE, MPI_DOUBLE,1,0,comm,&status);
			raw_time = finish - start - wtime_overhead - TS_OVERHEAD;
			printf("[%zu] Total: %f t_w: %lf\n", pass, raw_time, raw_time / (double) (BUFFER_SIZE * 4.0));
		}
	} else {
		for (size_t pass = 0; pass < MAX; pass++) {
			MPI_Barrier(comm);
			MPI_Recv(x, BUFFER_SIZE, MPI_DOUBLE, 0, 0,comm,&status);
			MPI_Send(x, BUFFER_SIZE, MPI_DOUBLE, 0, 0, comm);
		}
	}
	MPI_Finalize();
	return 0;
}

