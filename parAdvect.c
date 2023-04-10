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
#define CARTESIAN_HANDLERS

int M_loc, N_loc; // local advection field size (excluding halo) 
int M0, N0;       // local field element (0,0) is global element (M0,N0)
static int P0, Q0; // 2D process id (P0, Q0) in P x Q process grid 

static int M, N, P, Q; // local store of problem parameters
static int verbosity;
static int rank, nprocs;       // MPI values
static int topProc, botProc, leftProc, rightProc; // Neighbourhood proceses
static MPI_Comm comm;
static MPI_Comm commHandle;
static MPI_Datatype rowType;
static MPI_Datatype colType;

// Neighbourhood rank caluclation macros

#ifdef CARTESIAN_HANDLERS
#define calculateNeighbours() ({\
	if (P > 1) MPI_Cart_shift(commHandle, 0, -1, &topProc, &botProc); \
	if (Q > 1) MPI_Cart_shift(commHandle, 1, 1, &leftProc, &rightProc); \
})
#else
__attribute__((always_inline)) static inline int mod(int index, int axis) {
	if (index < 0) {
		return axis + index;
	} else if (index >= axis) {
		return index - axis;
	}
	return index;
}
#define calculateNeighbours() \
	if (P > 1) { \
		topProc = Q0 + (mod(P0 + 1, P) * Q); \
		botProc = Q0 + (mod(P0 - 1, P) * Q); \
	} \
	if (Q > 1) { \
		leftProc = mod(Q0 + 1, Q) + (P0 * Q); \
		rightProc = mod(Q0 - 1, Q) + (P0 * Q); \
	}

#endif

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

#ifdef CARTESIAN_HANDLERS
	int dimSize[2] = { P, Q };
	int periodicity[2] = { 1, 1 };
	MPI_Cart_create(comm, 2, dimSize, periodicity, 1, &commHandle);
#else
	commHandle = comm;
#endif
	calculateNeighbours();
} //initParParams()


void checkHaloSize(int w) {
	if (w > M_loc || w > N_loc) {
		printf("%d: w=%d too large for %dx%d local field! Exiting...\n",
				rank, w, M_loc, N_loc);
		exit(1);
	}
}

static void createRowColTypes(int haloWidth) {
	MPI_Type_vector(haloWidth, N_loc, N_loc + (haloWidth * 2), MPI_DOUBLE, &rowType);
	MPI_Type_vector(M_loc + (haloWidth * 2), haloWidth, N_loc + (haloWidth * 2), MPI_DOUBLE, &colType);
	MPI_Type_commit(&rowType);
	MPI_Type_commit(&colType);

}

// Exchange macros

#define MPI_Irow_exchange(srcY, srcX, srcRank, dstY, dstX, dstRank, index) \
	MPI_Irecv(&V(u, dstY, dstX), 1, rowType, dstRank, HALO_TAG, commHandle, &btRecvRequests[index]); \
	MPI_Isend(&V(u, srcY, srcX), 1, rowType, srcRank, HALO_TAG, commHandle, &requests[index])
#define MPI_Blocking_exchange(srcY, srcX, srcRank, dstY, dstX, dstRank, type) \
	MPI_Sendrecv( \
			&V(u, srcY, srcX), 1, type, srcRank, HALO_TAG, \
			&V(u, dstY, dstX), 1, type, dstRank, HALO_TAG, \
			commHandle, MPI_STATUS_IGNORE \
	)
#define MPI_Icol_exchange(srcY, srcX, srcRank, dstY, dstX, dstRank, index) \
		MPI_Irecv(&V(u, dstY, dstX), 1, colType, dstRank, HALO_TAG, commHandle, &requests[index]); \
		MPI_Isend(&V(u, srcY, srcX), 1, colType, srcRank, HALO_TAG, commHandle, &requests[(index) + 2])

static void updateBoundary(double *u, int ldu) {
	int i, j;

	//top and bottom halo 
	//note: we get the left/right neighbour's corner elements from each end
#ifdef HALO_NON_BLOCKING
	MPI_Request btRecvRequests[2];
	MPI_Request requests[6];
	size_t offset = 0;
#endif
	if (P == 1) {
		for (j = 1; j < N_loc+1; j++) {
			V(u, 0, j) = V(u, M_loc, j);
			V(u, M_loc+1, j) = V(u, 1, j);      
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(M_loc, 1, topProc, 0, 1, botProc, rowType);
		MPI_Blocking_exchange(1, 1, botProc, M_loc + 1, 1, topProc, rowType);
#else
		MPI_Irow_exchange(M_loc, 1, topProc, 0, 1, botProc, 0);
		MPI_Irow_exchange(1, 1, botProc, M_loc + 1, 1, topProc, 1);
		MPI_Waitall(2, btRecvRequests, NULL);
		offset = 2;
#endif
	}
	// left and right sides of halo
	if (Q == 1) { 
		for (i = 0; i < M_loc+2; i++) {
			V(u, i, 0) = V(u, i, N_loc);
			V(u, i, N_loc+1) = V(u, i, 1);
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(0, 1, leftProc, 0, N_loc + 1, rightProc, colType);
		MPI_Blocking_exchange(0, N_loc, rightProc, 0, 0, leftProc, colType);
#else
		MPI_Icol_exchange(0, 1, leftProc, 0, N_loc + 1, rightProc, offset);
		MPI_Icol_exchange(0, N_loc, rightProc, 0, 0, leftProc, offset + 1);
		offset = 6;
#endif
	}
#ifdef HALO_NON_BLOCKING
	MPI_Waitall(offset, requests, NULL);
#endif
} //updateBoundary()


// evolve advection over r timesteps, with (u,ldu) containing the local field
void parAdvect(int reps, double *u, int ldu) {
	int r; 
	double *v; int ldv = N_loc+2;
	v = calloc(ldv*(M_loc+2), sizeof(*v));
	assert(v != NULL);
	assert(ldu == N_loc + 2);
	createRowColTypes(1);

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
	if (Q > 1) {
		fprintf(stderr, "Overlapped comm/comp not supported for Q > 1");
		exit(1);
	}
	MPI_Request recvRequests[2];
	MPI_Request sendRequests[2];
	int ldv = N_loc + 2;
	double* v = calloc(ldv*(M_loc+2), sizeof(*v));
	assert(v != NULL);
	assert(ldu == N_loc + 2);
	createRowColTypes(1);
	// TODO: Fix this computation
	for (int r = 0; r < reps; r++) {
		// 1. Send ghost zones
		// Top and bottom of halo
		if (P == 1) {
			for (int j = 1; j < N_loc+1; j++) {
				V(u, 0, j) = V(u, M_loc, j);
				V(u, M_loc+1, j) = V(u, 1, j);      
			}
		} else {
			MPI_Irecv(&V(u, 0, 1), 1, rowType, botProc, HALO_TAG, commHandle, &recvRequests[0]);
			MPI_Irecv(&V(u, M_loc+1, 1), 1, rowType, topProc, HALO_TAG, commHandle, &recvRequests[1]);
			MPI_Isend(&V(u, M_loc, 1), 1, rowType, topProc, HALO_TAG, commHandle, &sendRequests[0]);
			MPI_Isend(&V(u, 1, 1), 1, rowType, botProc, HALO_TAG, commHandle, &sendRequests[1]);
		}
		// Left and right sides of halo
		if (Q == 1) { 
			for (int i = 0; i < M_loc+2; i++) {
				V(u, i, 0) = V(u, i, N_loc);
				V(u, i, N_loc+1) = V(u, i, 1);
			}
		}
		// 2. Compute advection for inner points
		updateAdvectField(M_loc - 2, N_loc, &V(u, 2, 1), ldu, &V(v, 2, 1), ldv);
		// 3. Wait for recieves
		MPI_Waitall(2, recvRequests, NULL);
		// 4. Compute advection for border points
		// Top
		updateAdvectField(1, N_loc, &V(u, M_loc + 1, 1), ldu, &V(v, M_loc + 1, 1), ldv);
		// Bottom
		updateAdvectField(1, N_loc, &V(u, 1, 1), ldu, &V(v, 1, 1), ldv);
		// 5. Wait for sends
		MPI_Waitall(2, sendRequests, NULL);
		// 6. Copy field
		copyField(M_loc, N_loc, &V(v, 1, 1), ldv, &V(u, 1, 1), ldu);
		if (verbosity > 2) {
			char s[64];
			sprintf(s, "%d reps: u", r+1);
			printAdvectField(rank, s, M_loc+2, N_loc+2, u, ldu);
		}
	}
	free(v);
} //parAdvectOverlap()



static void updateBoundaryWide(double *u, int ldu, int haloWidth) {
	int i, j;

	//top and bottom halo 
	//note: we get the left/right neighbour's corner elements from each end
#ifdef HALO_NON_BLOCKING
	MPI_Request btRecvRequests[2];
	MPI_Request requests[6];
	size_t offset = 0;
#endif
	if (P == 1) {
		for (j = 1; j < N_loc + haloWidth; j++) {
			V(u, 0, j) = V(u, M_loc, j);
			V(u, M_loc + haloWidth, j) = V(u, haloWidth, j);      
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(M_loc, haloWidth, topProc, 0, haloWidth, botProc, rowType);
		MPI_Blocking_exchange(haloWidth, haloWidth, botProc, M_loc + haloWidth, haloWidth, topProc, rowType);
#else
		MPI_Irow_exchange(M_loc, haloWidth, topProc, 0, haloWidth, botProc, 0);
		MPI_Irow_exchange(haloWidth, haloWidth, botProc, M_loc + haloWidth, haloWidth, topProc, 1);
		MPI_Waitall(2, btRecvRequests, NULL);
		offset = 2;
#endif
	}
	// left and right sides of halo
	if (Q == 1) { 
		for (i = 0; i < M_loc + haloWidth; i++) {
			V(u, i, 0) = V(u, i, N_loc);
			V(u, i, N_loc + haloWidth) = V(u, i, haloWidth);
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(0, haloWidth, leftProc, 0, N_loc + haloWidth, rightProc, colType);
		MPI_Blocking_exchange(0, N_loc, rightProc, 0, 0, leftProc, colType);
#else
		MPI_Icol_exchange(0, haloWidth, leftProc, 0, N_loc + haloWidth, rightProc, offset);
		MPI_Icol_exchange(0, N_loc, rightProc, 0, 0, leftProc, offset + 1);
		offset = 6;
#endif
	}
#ifdef HALO_NON_BLOCKING
	MPI_Waitall(offset, requests, NULL);
#endif
} //updateBoundary()

// wide halo variant
void parAdvectWide(int reps, int w, double *u, int ldu) {
	int r; 
	double *v; int ldv = N_loc + (w * 2);
	v = calloc(ldv*(M_loc + (w * 2)), sizeof(*v));
	assert(v != NULL);
	assert(ldu == N_loc + (w * 2));
	createRowColTypes(w);

	int updateIndex = 1;
	for (r = 0; r < reps; r++) {
		if (r % w == 0) {
			updateIndex = 1;
			updateBoundaryWide(u, ldu, w);
		}
		int m = M_loc + (2 * w) - (2 * updateIndex);
		int n = N_loc + (2 * w) - (2 * updateIndex);
		double* uOffset = &V(u, updateIndex, updateIndex);
		double* vOffset = &V(v, updateIndex, updateIndex);
		updateAdvectField(
				m, n,
				uOffset, ldu,
				vOffset, ldv
		);
		copyField(
				m, n,
				vOffset, ldv,
				uOffset, ldu
		);
		if (verbosity > 2) {
			char s[64]; sprintf(s, "%d reps: u", r+1);
			printAdvectField(rank, s, M_loc + (w * 2), N_loc + (w * 2), u, ldu);
		}
		updateIndex++;
	}
	free(v);

} //parAdvectWide()


// extra optimization variant
void parAdvectExtra(int r, double *u, int ldu) {

} //parAdvectExtra()
