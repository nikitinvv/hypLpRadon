/*interface*/
%module lpRgpu

%{
#define SWIG_FILE_WITH_INIT
#include "lpRgpu.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}


class lpRgpu{
	//global parameters
	int Nt,Nx,Nq,Ntau,ni;
	int Ntheta,Nrho;
	int Ntheta_R2C;
	int add;
	//gpu memoy
	float* dftx;
	float* dftauq;
	float* dfl;
	float2* dflc;
	float2* dfZ;
	int* dst;int* dstadj;int* dnvals;int* didnvals;
	
	int* didthetatx;int* didrhotx;
	int* didthetatauq;int* didrhotauq;
	float* ddthetatx;float* ddrhotx;
	float* ddthetatauq;float* ddrhotauq;
	float* demul;float* dcosmul;
	int* dreorids;int* dreoridsadj;
	float* dJ;
	//fft handles
	cufftHandle plan_forward;
	cufftHandle plan_inverse;
	cufftHandle plan_f_forward;
	cufftHandle plan_f_inverse;

	cudaError_t err;
	
public:
	//void callErr(const char* err);
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* N, int m0)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* fZ, int m1)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* st, int m2)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* stadj, int m3)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* idthetatx, int m4)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* idrhotx, int m5)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* idthetatauq, int m6)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* idrhotauq, int m7)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* dthetatx, int m8)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* drhotx, int m9)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* dthetatauq, int m10)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* drhotauq, int m11)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* emul, int m12)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* cosmul, int m13)};
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* J, int m14)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* reorids, int m15)};
%apply (int* INPLACE_ARRAY1, int DIM1) {(int* reoridsadj, int m16)};
	lpRgpu(int* N,int m0,float* fZ,int m1,int* st,int m2,int* stadj,int m3,int* idthetatx,int m4,int* idrhotx,int m5,int* idthetatauq,int m6,int* idrhotauq,int m7,float* dthetatx,int m8,float* drhotx,int m9,float* dthetatauq,int m10,float* drhotauq,int m11,float* emul,int m12,float* cosmul,int m13,float* J,int m14,int* reorids,int m15,int* reoridsadj,int m16);
	~lpRgpu();
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float* out, int os1, int os2)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* in, int is1,int is2)};
	void fwd(float* out, int os1, int os2, float* in, int is1, int is2);
	void adj(float* out, int os1, int os2, float* in, int is1, int is2);
%clear (float* in, int is1, int is1);
%clear (float* out, int os1, int os2);

	void printCurrentGPUMemory(const char* str);
	void fftlp(float* out,float* in);
	void fftlpadj(float* out,float* in);
	void convtx(float* out,float* in);
	void convtauq(float* out,float* in);
	void convtauqadj(float* out,float* in);
	void convtxadj(float* out,float* in);
	void getSizes(size_t *N);

};
