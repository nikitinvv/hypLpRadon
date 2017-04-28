#include <cufft.h>
#include <stdio.h>


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
	lpRgpu(int* N,int m0,float* fZ,int m1,int* st,int m2,int* stadj,int m3,int* idthetatx,int m4,int* idrhotx,int m5,int* idthetatauq,int m6,int* idrhotauq,int m7,float* dthetatx,int m8,
		float* drhotx,int m9,float* dthetatauq,int m10,float* drhotauq,int m11,float* emul,int m12,float* cosmul,int m13,float* J,int m14,int* reorids,int m15,int* reoridsadj,int m16);

	~lpRgpu();
	void fwd(float* out, int os1, int os2, float* in, int is1, int is2);
	void adj(float* out, int os1, int os2, float* in, int is1, int is2);
	void printCurrentGPUMemory(const char* str);
	void fftlp(float* out,float* in);
	void fftlpadj(float* out,float* in);
	void convtx(float* out,float* in);
	void convtauq(float* out,float* in);
	void convtauqadj(float* out,float* in);
	void convtxadj(float* out,float* in);
	void getSizes(size_t *N);

};
