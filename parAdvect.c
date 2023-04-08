// parallel 2D advection solver module
// written for COMP4300/8300 Assignment 1 
// v1.0 25 Feb 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#include "serAdvect.h"

#define HALO_TAG 100
#define HALO_NON_BLOCKING

int M_loc, N_loc; // local advection field size (excluding halo) 
int M0, N0;       // local field element (0,0) is global element (M0,N0)
static int P0, Q0; // 2D process id (P0, Q0) in P x Q process grid 

static int M, N, P, Q; // local store of problem parameters
static int verbosity;
static int rank, nprocs;       // MPI values
static MPI_Comm comm;
static MPI_Datatype rowType;
static MPI_Datatype colType;

//sets up parallel parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
	M = M_, N = N_; P = P_, Q = Q_;
	verbosity = verb;
	comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	P0 = rank / Q;
	M0 = (M / P) * P0;
	M_loc = (P0 < P - 1) ? (M / P) : (M - M0);

	assert(Q > 0);
	Q0 = rank % Q;
	N0 = (N / Q) * Q0;
	N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);

	MPI_Type_contiguous(M_loc, MPI_DOUBLE, &rowType);
	MPI_Type_vector(N_loc, 1, M_loc, MPI_DOUBLE, &colType);
	MPI_Type_commit(&rowType);
	MPI_Type_commit(&colType);
} //initParParams()


void checkHaloSize(int w) {
	if (w > M_loc || w > N_loc) {
		printf("%d: w=%d too large for %dx%d local field! Exiting...\n",
				rank, w, M_loc, N_loc);
		exit(1);
	}
}

static int mod(int index, int axis) {
	if (index < 0) {
		return axis + index;
	} else if (index >= axis) {
		return index - axis;
	}
	return index;
}

static void updateBoundary(double *u, int ldu) {
	int i, j;

	//top and bottom halo 
	//note: we get the left/right neighbour's corner elements from each end
#ifdef HALO_NON_BLOCKING
	MPI_Request requests[8];
	int offset = 0;
#endif
	if (P == 1) {
		for (j = 1; j < N_loc+1; j++) {
			V(u, 0, j) = V(u, M_loc, j);
			V(u, M_loc+1, j) = V(u, 1, j);      
		}
	} else {
		int topProc = Q0 + (mod(P0 + 1, P) * Q);
		int botProc = Q0 + (mod(P0 - 1, P) * Q);
#ifndef HALO_NON_BLOCKING
		if (rank % 2 == 0) {
			MPI_Recv(&V(u, 0, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
			MPI_Send(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm);
			MPI_Recv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
			MPI_Send(&V(u, 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm);
		} else {
			MPI_Send(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm);
			MPI_Recv(&V(u, 0, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
			MPI_Send(&V(u, 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm);
			MPI_Recv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
		}
#else
		MPI_Irecv(&V(u, 0, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &requests[0]);
		MPI_Irecv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &requests[1]);
		MPI_Isend(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &requests[2]);
		MPI_Isend(&V(u, 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &requests[3]);
		offset = 4;
#endif
	}
	// left and right sides of halo
	if (Q == 1) { 
		for (i = 0; i < M_loc+2; i++) {
			V(u, i, 0) = V(u, i, N_loc);
			V(u, i, N_loc+1) = V(u, i, 1);
		}
	} else {
		// FIXME: With Q > 1, the error rates go from e-06 to e-01. Might have to do with data type? Need to fix the errors.
		int leftProc = mod(Q0 + 1, Q) + (P0 * Q);
		int rightProc = mod(Q0 - 1, Q) + (P0 * Q);
#ifndef HALO_NON_BLOCKING
		if (rank % 2 == 0) {
			MPI_Recv(&V(u, 1, N_loc + 1), 1, colType, rightProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
			MPI_Send(&V(u, 1, 1), 1, colType, leftProc, HALO_TAG, comm);
			MPI_Recv(&V(u, 1, 0), 1, colType, leftProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
			MPI_Send(&V(u, 1, N_loc), 1, colType, rightProc, HALO_TAG, comm);
		} else {
			MPI_Send(&V(u, 1, 1), 1, colType, leftProc, HALO_TAG, comm);
			MPI_Recv(&V(u, 1, N_loc + 1), 1, colType, rightProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
			MPI_Send(&V(u, 1, N_loc), 1, colType, rightProc, HALO_TAG, comm);
			MPI_Recv(&V(u, 1, 0), 1, colType, leftProc, HALO_TAG, comm, MPI_STATUS_IGNORE);
		}
#else
		MPI_Irecv(&V(u, 1, N_loc + 1), 1, colType, rightProc, HALO_TAG, comm, &requests[offset + 0]);
		MPI_Irecv(&V(u, 1, 0), 1, colType, leftProc, HALO_TAG, comm, &requests[offset + 1]);
		MPI_Isend(&V(u, 1, 1), 1, colType, leftProc, HALO_TAG, comm, &requests[offset + 2]);
		MPI_Isend(&V(u, 1, N_loc), 1, colType, rightProc, HALO_TAG, comm, &requests[offset + 3]);
#endif
	}
#ifdef HALO_NON_BLOCKING
	MPI_Waitall(4 + offset, requests, NULL);
#endif
} //updateBoundary()


// evolve advection over r timesteps, with (u,ldu) containing the local field
void parAdvect(int reps, double *u, int ldu) {
	int r; 
	double *v; int ldv = N_loc+2;
	v = calloc(ldv*(M_loc+2), sizeof(*v));
	assert(v != NULL);
	assert(ldu == N_loc + 2);

	for (r = 0; r < reps; r++) {
		updateBoundary(u, ldu);
		updateAdvectField(M_loc, N_loc, &V(u,1,1), ldu, &V(v,1,1), ldv);
		copyField(M_loc, N_loc, &V(v,1,1), ldv, &V(u,1,1), ldu);

		if (verbosity > 2) {
			char s[64]; sprintf(s, "%d reps: u", r+1);
			printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
		}
	}

	free(v);
} //parAdvect()


// overlap communication variant
void parAdvectOverlap(int reps, double *u, int ldu) {
	if (Q != 1) {
		fprintf(stderr, "Overlapped advection is not supported for dimension Q != 1\n");
		exit(1);
	}
	MPI_Request requests[8];
	int offset = 0;
	int ldv = N_loc + 2;
	double* v = calloc(ldv*(M_loc+2), sizeof(*v));
	assert(v != NULL);
	assert(ldu == N_loc + 2);

	for (int r = 0; r < reps; r++) {
		// NOTE: Next send/recv updates

		// Top and bottom of halo
		if (P == 1) {
			for (int j = 1; j < N_loc+1; j++) {
				V(u, 0, j) = V(u, M_loc, j);
				V(u, M_loc+1, j) = V(u, 1, j);      
			}
		} else {
			int topProc = Q0 + (mod(P0 + 1, P) * Q);
			int botProc = Q0 + (mod(P0 - 1, P) * Q);
			MPI_Irecv(&V(u, 0, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &requests[0]);
			MPI_Irecv(&V(u, M_loc+1, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &requests[1]);
			MPI_Isend(&V(u, M_loc, 1), N_loc, MPI_DOUBLE, topProc, HALO_TAG, comm, &requests[2]);
			MPI_Isend(&V(u, 1, 1), N_loc, MPI_DOUBLE, botProc, HALO_TAG, comm, &requests[3]);
			offset = 4;
		}
		// Left and right sides of halo
		if (Q == 1) { 
			for (int i = 0; i < M_loc+2; i++) {
				V(u, i, 0) = V(u, i, N_loc);
				V(u, i, N_loc+1) = V(u, i, 1);
			}
		} else {
			int leftProc = mod(Q0 + 1, Q) + (P0 * Q);
			int rightProc = mod(Q0 - 1, Q) + (P0 * Q);
			MPI_Irecv(&V(u, 1, N_loc + 1), 1, colType, rightProc, HALO_TAG, comm, &requests[offset + 0]);
			MPI_Irecv(&V(u, 1, 0), 1, colType, leftProc, HALO_TAG, comm, &requests[offset + 1]);
			MPI_Isend(&V(u, 1, 1), 1, colType, leftProc, HALO_TAG, comm, &requests[offset + 2]);
			MPI_Isend(&V(u, 1, N_loc), 1, colType, rightProc, HALO_TAG, comm, &requests[offset + 3]);
		}

		// NOTE: Perform advection computation
		updateAdvectField(M_loc, N_loc, &V(u,1,1), ldu, &V(v,1,1), ldv);
		
		// NOTE: Wait on all send/recv before copying
		MPI_Waitall(4 + offset, requests, NULL);
		copyField(M_loc, N_loc, &V(v, 1, 1), ldv, &V(u, 1, 1), ldu);
		if (verbosity > 2) {
			char s[64];
			sprintf(s, "%d reps: u", r+1);
			printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
		}
	}
	free(v);
} //parAdvectOverlap()


// wide halo variant
void parAdvectWide(int reps, int w, double *u, int ldu) {

} //parAdvectWide()


// extra optimization variant
void parAdvectExtra(int r, double *u, int ldu) {

} //parAdvectExtra()
