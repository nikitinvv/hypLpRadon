#include "mex.h"
#include "class_handle.hpp"
#include "lpRgpu.cuh"
// The class that we are interfacing to
#include<iostream>
using namespace std;
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	// Get the command string
	char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

	// New
	
	if (!strcmp("new", cmd)) {
		// Check parameters
		if (nlhs != 1 || nrhs!=18)
			mexErrMsgTxt("New: Unexpected arguments.");
		// Return a handle to a new C++ instance
		/*	lpRgpu(int* N,float* fZ,int* st,int* stadj,int* idthetatx,int* idrhotx,int* idthetatauq,int* idrhotauq,float* dthetatx,
		float* drhotx,float* dthetatauq,float* drhotauq,float* emul,float* cosmul,float* J,int* reorids,int* reoridsadj);*/
		lpRgpu* lpRgpu0=new lpRgpu((int*)mxGetData(prhs[1]),(float*)mxGetData(prhs[2]),(int*)  mxGetData(prhs[3]),(int*)mxGetData(prhs[4]),
								   (int*)mxGetData(prhs[5]),(int*)mxGetData(prhs[6]),(int*)mxGetData(prhs[7]),(int*)mxGetData(prhs[8]),
								   (float*)mxGetData(prhs[9]),(float*)mxGetData(prhs[10]),(float*)mxGetData(prhs[11]),(float*)mxGetData(prhs[12]),
								   (float*)mxGetData(prhs[13]),(float*)mxGetData(prhs[14]),(float*)mxGetData(prhs[15]),(int*)mxGetData(prhs[16]),(int*)mxGetData(prhs[17]));
		//int* N,float* fZ,int* st, int* nvals,int* idthetatauq,int* idrhoadj,float* dthetatx,float* drhotx,float* dthetatauq,float* drhotauq
		plhs[0] = convertPtr2Mat<lpRgpu>(lpRgpu0);
		return;
	}

	// Check there is a second input, which should be the class instance handle
	if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");

	if (!strcmp("delete", cmd)) {
		// Destroy the C++ object
		destroyObject<lpRgpu>(prhs[1]);
		if (nlhs != 0 || nrhs != 2)
			mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
		return;
	}

	// Get the class instance pointer from the second input
	lpRgpu *lpRgpu0 = convertMat2Ptr<lpRgpu>(prhs[1]);
	if (!strcmp("fftlp", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("fftlp: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[4];Ns[1]=N[5];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->fftlp((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("fftlpadj", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("fftlp: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[4];Ns[1]=N[5];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->fftlpadj((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("convtx", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("convtx: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[4];Ns[1]=N[5];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->convtx((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("convtxadj", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("convtxadj: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[0];Ns[1]=N[1];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->convtxadj((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("convtauq", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("convtauq: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[2];Ns[1]=N[3];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->convtauq((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("convtauqadj", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("convtauqadj: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[4];Ns[1]=N[5];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->convtauqadj((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("fwd", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("fwd: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[2];Ns[1]=N[3];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->fwd((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}
	if (!strcmp("adj", cmd)) {
		if (nlhs > 1 || nrhs != 3)
			mexErrMsgTxt("adj: Unexpected arguments.");
		//size in
		size_t N[7];
		lpRgpu0->getSizes(N);size_t Ns[3];Ns[0]=N[0];Ns[1]=N[1];Ns[2]=N[6];
		plhs[0] = mxCreateNumericArray(3, Ns, mxSINGLE_CLASS, mxREAL);
		lpRgpu0->adj((float*)mxGetData(plhs[0]),(float*)mxGetData(prhs[2]));
		return;
	}


	// Got here, so command not recognized
	mexErrMsgTxt("Command not recognized.");
}
