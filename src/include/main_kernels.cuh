#define CONST1 0.166666666666667f
#define CONST2 0.666666666666667f
typedef float weights(float);
weights w0,w1,w2,w3;
__device__ weights *w[4]={w0,w1,w2,w3};
__device__ __inline__ float w0(float x){x=1.0f-x; return CONST1*x*x*x;}
__device__ __inline__ float w1(float x){x=x*x*(2.f-x); return fmaf(-0.5f,x,CONST2);}
__device__ __inline__ float w2(float x){x=(1.f-x)*(1.f-x)*(1.f+x);return fmaf(-0.5f,x,CONST2);}
__device__ __inline__ float w3(float x){return CONST1*x*x*x;}

__global__ void convtx_ker(float* out, float* in, int* st, float* dthetaa, float* drhoa, int* reorids, int Nt, int Nx, int Ntheta, int Nrho, int ni)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tz = blockIdx.z*blockDim.z + threadIdx.z;
	if(tx<Ntheta/4-2||tx>=3*Ntheta/4+2||ty>=Nrho||tz>=ni) return;
	int indg=tz*(Ntheta+4)*(Nrho+4)+(ty+2)*(Ntheta+4)+(tx+2);
	float res=0.0f;
	
	for (int j=0;j<4;j++)
	{
		for (int k=0;k<4;k++)
		{
			int mm=tx+k+(ty+j)*(Ntheta+4);
			int id1=st[mm];int id2=st[mm+1];
			for (int n=id1;n<id2;n++)
			{				
				int id=tz*Nt*Nx+reorids[n];//-1 matlab
				res +=  in[id] * w[3 - k](dthetaa[n])*w[3 - j](drhoa[n]);
			}
		}
	}
	out[indg]=res;
}
__global__ void convtauq_ker(float* out, float* in, int* idtheta, int* idrho, float* dthetaa, float* drhoa, int* reorids, int Nq, int Ntau, int Ntheta, int Nrho, int ni)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx >= Nq || ty >= Ntau || tz >= ni) return;
	int mm = ty*Nq + tx;
	int idthetal = idtheta[mm];
	int idrhol = idrho[mm];
	float dtheta = dthetaa[mm];
	float drho = drhoa[mm];
	int id = reorids[ty*Nq + tx];//-1 matlab
	int indg = tz*Nq*Ntau + id;
	float res = 0;
	tz *= (Ntheta + 4)*(Nrho + 4);
	for (int k = -1; k<3; k++)
		for (int j = -1; j<3; j++)
		{
			res += in[tz + idthetal + k + (Ntheta + 4)*(idrhol + j)]*w[1 + k](dtheta)*w[1 + j](drho);
		}
	out[indg] = res;
}
