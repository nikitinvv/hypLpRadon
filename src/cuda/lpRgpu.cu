#include "lpRgpu.cuh"
#include "main_kernels.cuh"
#include "simple_kernels.cuh"
#include "callerr.cuh"
lpRgpu::lpRgpu(int* N,int m0,float* fZ,int m1,int* st,int m2,int* stadj,int m3,int* idthetatx,int m4,int* idrhotx,int m5,int* idthetatauq,int m6,int* idrhotauq,int m7,float* dthetatx,int m8,
		float* drhotx,int m9,float* dthetatauq,int m10,float* drhotauq,int m11,float* emul,int m12,float* cosmul,int m13,float* J,int m14,int* reorids,int m15,int* reoridsadj,int m16)
{
	Nt=N[0];Nx=N[1];Nq=N[2];Ntau=N[3];Ntheta=N[4];Nrho=N[5];ni=N[6];
	Ntheta_R2C=(int)(Ntheta/2.0)+1;
	add=2;
	err=cudaMalloc((void **)&dftx, Nt*Nx*ni*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dftauq, Ntau*Nq*ni*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dfl, (Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dflc, Ntheta_R2C*Nrho*ni*sizeof(float2));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dfZ, Ntheta_R2C*Nrho*sizeof(float2));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dst, (Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dstadj, (Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	
	err=cudaMalloc((void **)&ddthetatx, Nt*Nx*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&ddrhotx, Nt*Nx*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&ddthetatauq, Ntau*Nq*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&ddrhotauq, Ntau*Nq*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));

	err=cudaMalloc((void **)&didthetatx, Nt*Nx*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&didrhotx, Nt*Nx*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&didthetatauq, Nq*Ntau*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&didrhotauq, Nq*Ntau*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	
	err=cudaMalloc((void **)&demul, Nt*Nx*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dcosmul, Ntau*Nq*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dJ, Nt*Nx*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dreorids, Nt*Nx*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dreoridsadj, Ntau*Nq*sizeof(int));if (err!=0) callErr(cudaGetErrorString(err));

	err=cudaMemcpy(ddthetatx,dthetatx, Nt*Nx*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(ddrhotx,drhotx, Nt*Nx*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(ddthetatauq,dthetatauq, Ntau*Nq*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(ddrhotauq,drhotauq, Ntau*Nq*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));

	err=cudaMemcpy(dst,st,(Ntheta+2*add)*(Nrho+2*add)*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dstadj,stadj,(Ntheta+2*add)*(Nrho+2*add)*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));


	err=cudaMemcpy(dfZ,fZ,Ntheta_R2C*Nrho*sizeof(float2),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(didthetatx,idthetatx,Nt*Nx*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(didrhotx,idrhotx,Nt*Nx*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(didthetatauq,idthetatauq,Nq*Ntau*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(didrhotauq,idrhotauq,Nq*Ntau*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));

	err=cudaMemcpy(demul,emul,Nt*Nx*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dcosmul,cosmul,Nq*Ntau*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dJ,J,Nt*Nx*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dreorids,reorids,Nt*Nx*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemcpy(dreoridsadj,reoridsadj,Ntau*Nq*sizeof(int),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));

	//fft plans for ni slices
	cufftResult res1,res2;
	int ffts[]={Nrho,Ntheta};
	int idist = (Nrho+2*add)*(Ntheta+2*add);int odist = (Nrho)*((Ntheta/2+1));
	int inembed[] = {Nrho+2*add, Ntheta+2*add};int onembed[] = {Nrho, Ntheta/2+1};
	res1=cufftPlanMany(&plan_forward, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, ni);if (res1!=0) {char errs[16];sprintf(errs,"fwd cufftPlanMany error %d",res1);callErr(errs);}
	res2=cufftPlanMany(&plan_inverse, 2, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2R, ni);if (res2!=0) {char errs[16];sprintf(errs,"inv cufftPlanMany error %d",res1);callErr(errs);}
}
lpRgpu::~lpRgpu()
{
	cudaFree(dftx);
	cudaFree(dftauq);
	cudaFree(dfl);
	cudaFree(dflc);
	cudaFree(dfZ);

	cudaFree(dst);
	cudaFree(dstadj);

	cudaFree(ddthetatx);
	cudaFree(ddrhotx);
	cudaFree(ddthetatauq);
	cudaFree(ddrhotauq);

	cudaFree(didthetatx);
	cudaFree(didrhotx);
	cudaFree(didthetatauq);
	cudaFree(didrhotauq);

	cudaFree(demul);
	cudaFree(dcosmul);
	cudaFree(dJ);
	cudaFree(dreorids);
	cudaFree(dreoridsadj);
	
	cufftDestroy(plan_forward);
	cufftDestroy(plan_inverse);
}
void lpRgpu::getSizes(size_t *N)
{
	N[0]=Nt;N[1]=Nx;N[2]=Nq;N[3]=Ntau;N[4]=Ntheta;N[5]=Nrho;N[6]=ni;
}
void lpRgpu::fwd(float* out, int os1, int os2, float* in, int is1, int is2)
{
	err=cudaMemcpy(dftx,in,Nt*Nx*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
	cudaMemset(dfl,0,(Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float));

	int MBS31,MBS32,MBS33; MBS31=16;MBS32=16;MBS33=4;
	dim3 dimBlock(MBS31,MBS32,MBS33);	
	int MBS31g,MBS32g,MBS33g; MBS31g=4;MBS32g=4;MBS33g=64;
	dim3 dimBlockg(MBS31g,MBS32g,MBS33g);	
	
	
//mul erho,J
	int GS31=(int)ceil(Nt/(float)MBS31);int GS32=(int)ceil(Nx/(float)MBS32);int GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid0(GS31,GS32,GS33);	
	pmul <<<dimGrid0,dimBlock>>>(dftx,dJ,Nt,Nx,ni);cudaDeviceSynchronize();
	pmul <<<dimGrid0,dimBlock>>>(dftx,demul,Nt,Nx,ni);cudaDeviceSynchronize();	


//convtx
	GS31=(int)ceil(Ntheta/(float)MBS31g);GS32=(int)ceil(Nrho/(float)MBS32g);GS33=(int)ceil(ni/(float)MBS33g);dim3 dimGrid1(GS31,GS32,GS33);	
	convtx_ker<<<dimGrid1,dimBlockg>>>(dfl,dftx,dst,ddthetatx,ddrhotx,dreorids,Nt,Nx,Ntheta,Nrho,ni);cudaDeviceSynchronize();

	
//fftlp
	cufftExecR2C(plan_forward, (cufftReal*)&dfl[(Ntheta+2*add)*add+add],(cufftComplex*)dflc);cudaDeviceSynchronize();
	GS31=(int)ceil(Ntheta_R2C/(float)MBS31);GS32=(int)ceil(Nrho/(float)MBS32);GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid2(GS31,GS32,GS33);
	mul<<<dimGrid2, dimBlock>>>(1/(float)(Ntheta*Nrho),dflc,dfZ,Ntheta_R2C,Nrho,ni);cudaDeviceSynchronize();
	cufftExecC2R(plan_inverse,(cufftComplex*)dflc,(cufftReal*)&dfl[(Ntheta+2*add)*add+add]);cudaDeviceSynchronize();	

//convtauq
	GS31=(int)ceil(Nq/(float)MBS31);GS32=(int)ceil(Ntau/(float)MBS32);GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid3(GS31,GS32,GS33);	
	convtauq_ker<<<dimGrid3,dimBlock>>>(dftauq,dfl,didthetatauq,didrhotauq,ddthetatauq,ddrhotauq,dreoridsadj,Nq,Ntau,Ntheta,Nrho,ni);cudaDeviceSynchronize();		

//mul cos
	pmul <<<dimGrid3,dimBlock>>>(dftauq,dcosmul,Nq,Ntau,ni);cudaDeviceSynchronize();


	err=cudaMemcpy(out,dftauq,Nq*Ntau*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
}
void lpRgpu::adj(float* out, int os1, int os2, float* in, int is1, int is2)
{
	err=cudaMemcpy(dftauq,in,Ntau*Nq*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	


	int MBS31,MBS32,MBS33; MBS31=16;MBS32=16;MBS33=4;
	dim3 dimBlock(MBS31,MBS32,MBS33);	
	int MBS31g,MBS32g,MBS33g; MBS31g=4;MBS32g=4;MBS33g=64;
	dim3 dimBlockg(MBS31g,MBS32g,MBS33g);	

	//mul cos
	int GS31=(int)ceil(Nq/(float)MBS31);int GS32=(int)ceil(Ntau/(float)MBS32);int GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid0(GS31,GS32,GS33);	
	pmul <<<dimGrid0,dimBlock>>>(dftauq,dcosmul,Nq,Ntau,ni);cudaDeviceSynchronize();
	//conv
	cudaMemset(dfl,0,(Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float));	
	GS31=(int)ceil(Ntheta/(float)MBS31g);GS32=(int)ceil(Nrho/(float)MBS32g);GS33=(int)ceil(ni/(float)MBS33g);dim3 dimGrid1(GS31,GS32,GS33);	
	convtx_ker<<<dimGrid1,dimBlockg>>>(dfl,dftauq,dstadj,ddthetatauq,ddrhotauq,dreoridsadj,Nq,Ntau,Ntheta,Nrho,ni);cudaDeviceSynchronize();
	//fftlp
	cufftExecR2C(plan_forward, (cufftReal*)dfl,(cufftComplex*)dflc);	
	GS31=(int)ceil(Ntheta_R2C/(float)MBS31);GS32=(int)ceil(Nrho/(float)MBS32);GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid2(GS31,GS32,GS33);
	muladj<<<dimGrid2, dimBlock>>>(1/(float)(Ntheta*Nrho),dflc,dfZ,Ntheta_R2C,Nrho,ni);cudaDeviceSynchronize();
	cufftExecC2R(plan_inverse,(cufftComplex*)dflc,(cufftReal*)dfl);cudaDeviceSynchronize();
	//conv	
	GS31=(int)ceil(Nt/(float)MBS31);GS32=(int)ceil(Nx/(float)MBS32);GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid3(GS31,GS32,GS33);	
	convtauq_ker<<<dimGrid3,dimBlock>>>(dftx,dfl,didthetatx,didrhotx,ddthetatx,ddrhotx,dreorids,Nt,Nx,Ntheta,Nrho,ni);cudaDeviceSynchronize();
	//mul erho,J
	pmul <<<dimGrid3,dimBlock>>>(dftx,dJ,Nt,Nx,ni);cudaDeviceSynchronize();
	pmul <<<dimGrid3,dimBlock>>>(dftx,demul,Nt,Nx,ni);cudaDeviceSynchronize();	

	err=cudaMemcpy(out,dftx,Nt*Nx*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
}

void lpRgpu::fftlp(float* out,float* in)
{
	err=cudaMemcpy(dfl,in,Ntheta*Nrho*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	//forward FFT
	cufftExecR2C(plan_forward, (cufftReal*)dfl,(cufftComplex*)dflc);
	//multiplication by fZ
		int MBS31,MBS32,MBS33; MBS31=8;MBS32=8;MBS33=16;
	dim3 dimBlock(MBS31,MBS32,MBS33);	
	int GS31=(int)ceil(Ntheta_R2C/(float)MBS31);int GS32=(int)ceil(Nrho/(float)MBS32);int GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid(GS31,GS32,GS33);
	mul<<<dimGrid, dimBlock>>>(1/(float)(Ntheta*Nrho),dflc,dfZ,Ntheta_R2C,Nrho,ni);cudaDeviceSynchronize();
	//inverse FFT
	cufftExecC2R(plan_inverse,(cufftComplex*)dflc,(cufftReal*)dfl);
	cudaDeviceSynchronize();

	err=cudaMemcpy(out,dfl,Ntheta*Nrho*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
}
void lpRgpu::fftlpadj(float* out,float* in)
{
	//forward FFT	
	cufftExecR2C(plan_forward, (cufftReal*)dfl,(cufftComplex*)dflc);
	//multiplication by fZ
	int MBS31,MBS32,MBS33; MBS31=8;MBS32=8;MBS33=16;
	dim3 dimBlock(MBS31,MBS32,MBS33);	
	int GS31=(int)ceil(Ntheta_R2C/(float)MBS31);int GS32=(int)ceil(Nrho/(float)MBS32);int GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid(GS31,GS32,GS33);
	muladj<<<dimGrid, dimBlock>>>(1/(float)(Ntheta*Nrho),dflc,dfZ,Ntheta_R2C,Nrho,ni);cudaDeviceSynchronize();
	//inverse FFT
	cufftExecC2R(plan_inverse,(cufftComplex*)dflc,(cufftReal*)dfl);
	cudaDeviceSynchronize();	
	err=cudaMemcpy(out,dfl,Ntheta*Nrho*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
}
void lpRgpu::convtx(float* out,float* in)
{
	err=cudaMemcpy(dftx,in,Nt*Nx*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
	cudaMemset(dfl,0,(Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float));
	int MBS31_,MBS32_,MBS33_;
	MBS31_=32;MBS32_=32;MBS33_=1;
	dim3 dimBlock(MBS31_,MBS32_,MBS33_);	
	int GS31=(int)ceil(Ntheta/(float)MBS31_);int GS32=(int)ceil(Nrho/(float)MBS32_);int GS33=(int)ceil(ni/(float)MBS33_);dim3 dimGrid(GS31,GS32,GS33);	
	convtx_ker<<<dimGrid,dimBlock>>>(dfl,dftx,dst,ddthetatx,ddrhotx,dreorids,Nt,Nx,Ntheta,Nrho,ni);cudaDeviceSynchronize();
	err=cudaMemcpy(out,dfl,Ntheta*Nrho*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
}
void lpRgpu::convtauq(float* out,float* in)
{
	err=cudaMemcpy(dfl,in,(Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
	int MBS31,MBS32,MBS33; MBS31=8;MBS32=8;MBS33=16;
	dim3 dimBlock(MBS31,MBS32,MBS33);	
	int GS31=(int)ceil(Nq/(float)MBS31);int GS32=(int)ceil(Ntau/(float)MBS32);int GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid(GS31,GS32,GS33);	
	convtauq_ker<<<dimGrid,dimBlock>>>(dftauq,dfl,didthetatauq,didrhotauq,ddthetatauq,ddrhotauq,dreoridsadj,Nq,Ntau,Ntheta,Nrho,ni);cudaDeviceSynchronize();
	err=cudaMemcpy(out,dftauq,Nq*Ntau*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
}

void lpRgpu::convtauqadj(float* out,float* in)
{
	err=cudaMemcpy(dftauq,in,Ntau*Nq*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
	cudaMemset(dfl,0,(Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float));
	int MBS31_,MBS32_,MBS33_;
	MBS31_=8;MBS32_=8;MBS33_=16;
	dim3 dimBlock(MBS31_,MBS32_,MBS33_);	
	int GS31=(int)ceil(Ntheta/(float)MBS31_);int GS32=(int)ceil(Nrho/(float)MBS32_);int GS33=(int)ceil(ni/(float)MBS33_);dim3 dimGrid(GS31,GS32,GS33);	
	convtx_ker<<<dimGrid,dimBlock>>>(dfl,dftauq,dstadj,ddthetatauq,ddrhotauq,dreorids,Nq,Ntau,Ntheta,Nrho,ni);cudaDeviceSynchronize();
	err=cudaMemcpy(out,dfl,Ntheta*Nrho*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
}

void lpRgpu::convtxadj(float* out,float* in)
{
	err=cudaMemcpy(dfl,in,(Ntheta+2*add)*(Nrho+2*add)*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
		int MBS31,MBS32,MBS33; MBS31=8;MBS32=8;MBS33=16;
	dim3 dimBlock(MBS31,MBS32,MBS33);	
	int GS31=(int)ceil(Nt/(float)MBS31);int GS32=(int)ceil(Nx/(float)MBS32);int GS33=(int)ceil(ni/(float)MBS33);dim3 dimGrid(GS31,GS32,GS33);	
	convtauq_ker<<<dimGrid,dimBlock>>>(dftx,dfl,didthetatx,didrhotx,ddthetatx,ddrhotx,dreoridsadj,Nt,Nx,Ntheta,Nrho,ni);cudaDeviceSynchronize();
	err=cudaMemcpy(out,dftx,Nt*Nx*ni*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));	
}

void lpRgpu::printCurrentGPUMemory(const char* str)
{
	size_t gpufree1,gputotal;
	cudaMemGetInfo(&gpufree1,&gputotal);
	if(str!=NULL)
		printf("%s gpufree=%.0fM,gputotal=%.0fM\n",str,gpufree1/(float)(1024*1024),gputotal/(float)(1024*1024));
	else
		printf("gpufree=%.0fM,gputotal=%.0fM\n",gpufree1/(float)(1024*1024),gputotal/(float)(1024*1024));
}
