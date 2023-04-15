// parallel 2D advection solver module
// written for COMP4300/8300 Assignment 1 
// v1.0 25 Feb 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <memory.h>
#include <complex.h>
#include <math.h>

//#define FFT_CONV_KERNEL 0
#if FFT_CONV_KERNEL == 1
#include <fftw3.h>
#endif

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
static int topProc, botProc, leftProc, rightProc; // Von-Neumann neighbourhood proceses
static int topLeftProc, topRightProc, botLeftProc, botRightProc; // Corner neighbourhood processes
static MPI_Comm comm, commHandle; // Communication handlers for main and Cartesian configurations
static MPI_Datatype rowType, colType, cornerType; // Data types for exchanges

// Neighbourhood rank caluclation macros

#ifdef CARTESIAN_HANDLERS
#define calculateNeighbours() ({\
	if (P > 1) MPI_Cart_shift(commHandle, 0, -1, &topProc, &botProc); \
	if (Q > 1) MPI_Cart_shift(commHandle, 1, 1, &leftProc, &rightProc); \
	int _tlc[] = { P0 + 1, Q0 + 1 }; \
	int _trc[] = { P0 + 1, Q0 - 1 }; \
	int _blc[] = { P0 - 1, Q0 + 1 }; \
	int _brc[] = { P0 - 1, Q0 - 1 }; \
	MPI_Cart_rank(commHandle, _tlc, &topLeftProc); \
	MPI_Cart_rank(commHandle, _trc, &topRightProc); \
	MPI_Cart_rank(commHandle, _blc, &botLeftProc); \
	MPI_Cart_rank(commHandle, _brc, &botRightProc); \
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
#define coordShift(coords) mod(Q0 + (coords)[1], Q) + (mod(P0 + (coords)[0], P) * Q);
#define calculateNeighbours() \
	if (P > 1) { \
		topProc = Q0 + (mod(P0 + 1, P) * Q); \
		botProc = Q0 + (mod(P0 - 1, P) * Q); \
	} \
	if (Q > 1) { \
		leftProc = mod(Q0 + 1, Q) + (P0 * Q); \
		rightProc = mod(Q0 - 1, Q) + (P0 * Q); \
	} \
	int _tlc[] = { +1, +1 }; \
	int _trc[] = { +1, -1 }; \
	int _blc[] = { -1, +1 }; \
	int _brc[] = { -1, -1 }; \
	topLeftProc = coordShift(_tlc); \
	topRightProc = coordShift(_trc); \
	botLeftProc = coordShift(_blc); \
	botRightProc = coordShift(_brc);
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
	if (rank == 0 && (w > M_loc || w > N_loc)) {
		fprintf(
				stderr,
				"%d: w=%d too large for %dx%d local field! Exiting...\n",
				rank,
				w,
				M_loc,
				N_loc
		);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

static void createRowColTypes(int haloWidth) {
	MPI_Type_vector(haloWidth, N_loc, N_loc + (haloWidth * 2), MPI_DOUBLE, &rowType);
	MPI_Type_vector(M_loc, haloWidth, N_loc + (haloWidth * 2), MPI_DOUBLE, &colType);
	MPI_Type_vector(haloWidth, haloWidth, N_loc + (haloWidth * 2), MPI_DOUBLE, &cornerType);
	MPI_Type_commit(&rowType);
	MPI_Type_commit(&colType);
	MPI_Type_commit(&cornerType);

}

// Exchange macros
#define MPI_Init_exchange() \
	MPI_Request recvRequests[8]; \
	MPI_Request sendRequests[8]; \
	size_t MPI_Reset_exchange()

#define MPI_Reset_exchange() offset = 0

#define MPI_Waitall_exchange() \
	MPI_Waitall(offset, recvRequests, NULL); \
	MPI_Waitall(offset, sendRequests, NULL)

#define MPI_Irow_exchange(srcY, srcX, srcRank, dstY, dstX, dstRank) \
	MPI_Irecv(&V(u, dstY, dstX), 1, rowType, dstRank, HALO_TAG, commHandle, &recvRequests[offset]); \
	MPI_Isend(&V(u, srcY, srcX), 1, rowType, srcRank, HALO_TAG, commHandle, &sendRequests[offset]); \
	offset++

#define MPI_Blocking_exchange(srcY, srcX, srcRank, dstY, dstX, dstRank, type) \
	MPI_Sendrecv( \
			&V(u, srcY, srcX), 1, type, srcRank, HALO_TAG, \
			&V(u, dstY, dstX), 1, type, dstRank, HALO_TAG, \
			commHandle, MPI_STATUS_IGNORE \
	)

#define MPI_Icol_exchange(srcY, srcX, srcRank, dstY, dstX, dstRank) \
	MPI_Irecv(&V(u, dstY, dstX), 1, colType, dstRank, HALO_TAG, commHandle, &recvRequests[offset]); \
	MPI_Isend(&V(u, srcY, srcX), 1, colType, srcRank, HALO_TAG, commHandle, &sendRequests[offset]); \
	offset++

#define MPI_Icorner_exchange(srcY, srcX, dstY, dstX, rank) \
	MPI_Irecv(&V(u, dstY, dstX), 1, cornerType, rank, HALO_TAG, commHandle, &recvRequests[offset]); \
	MPI_Isend(&V(u, srcY, srcX), 1, cornerType, rank, HALO_TAG, commHandle, &sendRequests[offset]); \
	offset++

static void updateBoundary(double *u, int ldu) {
	//top and bottom halo 
	//note: we get the left/right neighbour's corner elements from each end
#ifdef HALO_NON_BLOCKING
	MPI_Init_exchange();
#endif
	if (P == 1) {
		for (int j = 1; j < N_loc+1; j++) {
			V(u, 0, j) = V(u, M_loc, j);
			V(u, M_loc+1, j) = V(u, 1, j);      
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(M_loc, 1, topProc, 0, 1, botProc, rowType);
		MPI_Blocking_exchange(1, 1, botProc, M_loc + 1, 1, topProc, rowType);
#else
		MPI_Irow_exchange(M_loc, 1, topProc, 0, 1, botProc);
		MPI_Irow_exchange(1, 1, botProc, M_loc + 1, 1, topProc);
#endif
	}
	// left and right sides of halo
	if (Q == 1) {
#ifdef HALO_NON_BLOCKING
		MPI_Waitall_exchange();
#endif
		for (int i = 0; i < M_loc + 2; i++) {
			V(u, i, 0) = V(u, i, N_loc);
			V(u, i, N_loc+1) = V(u, i, 1);
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(0, 1, leftProc, 0, N_loc + 1, rightProc, colType);
		MPI_Blocking_exchange(0, N_loc, rightProc, 0, 0, leftProc, colType);
#else
		MPI_Icol_exchange(1, 1, leftProc, 1, N_loc + 1, rightProc);
		MPI_Icol_exchange(1, N_loc, rightProc, 1, 0, leftProc);
		MPI_Icorner_exchange(1, 1, 0, 0, botLeftProc);
		MPI_Icorner_exchange(M_loc, N_loc, M_loc + 1, N_loc + 1, topRightProc);
		MPI_Icorner_exchange(1, N_loc, 0, N_loc + 1, botRightProc);
		MPI_Icorner_exchange(M_loc, 1, M_loc + 1, 0, topLeftProc);
		MPI_Waitall_exchange();
#endif
	}
} //updateBoundary()


// evolve advection over r timesteps, with (u,ldu) containing the local field
void parAdvect(int reps, double *u, int ldu) {
	int ldv = N_loc + 2;
	assert(ldu == ldv);
	double* v = calloc(ldv * ldv, sizeof(*v));
	assert(v != NULL);
	createRowColTypes(1);

	for (int r = 0; r < reps; r++) {
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
	int ldv = N_loc + 2;
	assert(ldu == ldv);
	double* v = calloc(ldv * ldv, sizeof(*v));
	assert(v != NULL);
	MPI_Init_exchange();
	createRowColTypes(1);
	for (int r = 0; r < reps; r++) {
		MPI_Reset_exchange();
		// 1. Send ghost zones
		// Top and bottom sides of halo (no corners)
		if (P == 1) {
			for (int j = 1; j < N_loc + 1; j++) {
				V(u, 0, j) = V(u, M_loc, j);
				V(u, M_loc+1, j) = V(u, 1, j);      
			}
		} else {
			MPI_Irow_exchange(M_loc, 1, topProc, 0, 1, botProc);
			MPI_Irow_exchange(1, 1, botProc, M_loc + 1, 1, topProc);
		}
		// Left and right sides of halo (no corners)
		if (Q == 1) { 
			for (int i = 1; i < M_loc + 1; i++) {
				V(u, i, 0) = V(u, i, N_loc);
				V(u, i, N_loc+1) = V(u, i, 1);
			}
		} else {
			MPI_Icol_exchange(1, 1, leftProc, 1, N_loc + 1, rightProc);
			MPI_Icol_exchange(1, N_loc, rightProc, 1, 0, leftProc);
			MPI_Icorner_exchange(1, 1, 0, 0, botLeftProc);
			MPI_Icorner_exchange(M_loc, N_loc, M_loc + 1, N_loc + 1, topRightProc);
			MPI_Icorner_exchange(1, N_loc, 0, N_loc + 1, botRightProc);
			MPI_Icorner_exchange(M_loc, 1, M_loc + 1, 0, topLeftProc);
		}
		// 2. Compute advection for inner points
		updateAdvectField(M_loc - 2, N_loc - 2, &V(u, 2, 2), ldu, &V(v, 2, 2), ldv);
		// 3. Wait for recieves
		MPI_Waitall(offset, recvRequests, NULL);
		if (Q == 1) {
			// Send corners
			V(u, 0, 0) = V(u, 0, N_loc);
			V(u, 0, N_loc + 1) = V(u, 0, 1);
			V(u, M_loc + 1, 0) = V(u, M_loc + 1, N_loc);
			V(u, M_loc + 1, N_loc + 1)  = V(u, M_loc + 1, 1);
		}
		// 4. Compute advection for border points
		// Top
		updateAdvectField(1, N_loc - 2, &V(u, M_loc, 2), ldu, &V(v, M_loc, 2), ldv);
		// Bottom
		updateAdvectField(1, N_loc - 2, &V(u, 1, 2), ldu, &V(v, 1, 2), ldv);
		// Left
		updateAdvectField(M_loc, 1, &V(u, 1, 1), ldu, &V(v, 1, 1), ldv);
		// Right
		updateAdvectField(M_loc, 1, &V(u, 1, N_loc), ldu, &V(v, 1, N_loc), ldv);
		// 5. Wait for sends
		MPI_Waitall(offset, sendRequests, NULL);
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



static void updateBoundaryWide(double *u, int ldu, int w) {
	//top and bottom halo 
	//note: we get the left/right neighbour's corner elements from each end
#ifdef HALO_NON_BLOCKING
	MPI_Init_exchange();
#endif
	if (P == 1) {
		for (int j = 1; j < N_loc + w; j++) {
			V(u, 0, j) = V(u, M_loc, j);
			V(u, M_loc + w, j) = V(u, w, j);      
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(M_loc, w, topProc, 0, w, botProc, rowType);
		MPI_Blocking_exchange(w, w, botProc, M_loc + w, w, topProc, rowType);
#else
		MPI_Irow_exchange(M_loc, w, topProc, 0, w, botProc);
		MPI_Irow_exchange(w, w, botProc, M_loc + w, w, topProc);
#endif
	}
	// left and right sides of halo
	if (Q == 1) { 
#ifdef HALO_NON_BLOCKING
		MPI_Waitall_exchange();
#endif
		for (int i = 0; i < M_loc + (w * 2); i++) {
			V(u, i, 0) = V(u, i, N_loc);
			V(u, i, N_loc + w) = V(u, i, w);
		}
	} else {
#ifndef HALO_NON_BLOCKING
		MPI_Blocking_exchange(0, w, leftProc, 0, N_loc + w, rightProc, colType);
		MPI_Blocking_exchange(0, N_loc, rightProc, 0, 0, leftProc, colType);
#else
		MPI_Icol_exchange(w, w, leftProc, w, N_loc + w, rightProc);
		MPI_Icol_exchange(w, N_loc, rightProc, w, 0, leftProc);
		MPI_Icorner_exchange(w, w, 0, 0, botLeftProc);
		MPI_Icorner_exchange(M_loc, N_loc, M_loc + w, N_loc + w, topRightProc);
		MPI_Icorner_exchange(w, N_loc, 0, N_loc + w, botRightProc);
		MPI_Icorner_exchange(M_loc, w, M_loc + w, 0, topLeftProc);
		MPI_Waitall_exchange();
#endif
	}
} //updateBoundary()

// wide halo variant
void parAdvectWide(int reps, int w, double *u, int ldu) {
	int ldv = N_loc + (w * 2);
	assert(ldu == ldv);
	double* v = calloc(ldv * ldv, sizeof(*v));
	assert(v != NULL);
	createRowColTypes(w);
	int updateIndex = 1;
	for (int r = 0; r < reps; r++) {
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

void _fft(double complex *restrict const buf,
		  double complex *restrict const out,
		  size_t const n,
		  size_t const step) {
	if (step >= n) {
		return;
	}
	double complex t;
	_fft(out, buf, n, step * 2);
	_fft(out + step, buf + step, n, step * 2);
	for (size_t i = 0; i < n; i+= 2 * step) {
		t = cexp(-I * M_PI * (double) i / (double) n) * out[i + step];
		buf[i / 2] = out[i] + t;
		buf[i + (n - i) / 2] = out[i] - t;
	}
}

int fft(double complex *const buf, const size_t n) {
	double complex out[n];
	if ((n & (n -1)) != 0) {
		return -1;
	}
	memcpy(out, buf, n);
	_fft(buf, out, n, 1);
	return 0;
}

// extra optimization variant
void parAdvectExtra(int r, double *u, int ldu) {
	// -------------------------
	// S = Stencil
	// a_0 = Initial data
	// a_T = Final data
	// F = DFT matrix
	// F^-1 = Inverse DFT matrix
	// FSF^-1 = Diagronal matrix
	// T = Number of timesteps
	// MP = Matrix product
	// EV = Evolution
	// IFFT = Inverse FFT
	// RS = Repeated squaring
	// -------------------------
	//
	// [S] -FFT-> [FSF^-1] -RS-> [F*S^T*F^-1]
	//                           |
	// [a_0] -FFT-> [F*a_0] -----+-MP--> [(F*S^T*F^-1)(F*a_0)] -EV-> [F*a_T] -IFFT-> [a_T]
#if FFT_CONV_KERNEL == 1
	fftw_complex* in;
	fftw_complex* out;
	fftw_plan p;
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (M_loc * N_loc));
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (M_loc * N_loc));
	p = fftw_plan_dft_1d(M_loc * N_loc, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
#endif
} //parAdvectExtra()
