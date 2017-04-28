from numpy import *
from lpRgpu import lpRgpu

class hypRadon:
	def __init__(self, t, x, q, tau, ni, Ntheta, Nrho):
		#normalize coordinates
		[Nx,Nt]=shape(t);[Ntau,Nq]=shape(q);
		T0=t[0,0];T1=t[-1,-1]+t[0,1]-t[0,0];X1=x[-1,-1]+x[1,0]-x[0,0];
		t=t/T1;x=x/X1;
		q=q*X1/T1;tau=tau/T1;
		#lp parameters
		q2=q**2;gamma=0;
		alpha=(max(ndarray.flatten(arctan(q2)))+min(ndarray.flatten(arctan(q2))))/2;

		beta=max(ndarray.flatten(arctan(q2)))-min(ndarray.flatten(arctan(q2)));
		aR=sqrt(2)*sin(beta)*sqrt(1/(2 + 2*sin(2*alpha)*(cos(beta) + sin(beta)) + sin(2*beta)));
		Ox= -sqrt(0.2e1) * sin(alpha + pi / 0.4e1) * sin(beta) * aR / (-0.1e1 + cos(beta)) / 0.2e1;
		Oy= sqrt(0.2e1) * aR * cos(alpha + pi / 0.4e1) * (0.1e1 - cos(beta)) / sin(beta) / 0.2e1;
		recx=array([-0.5,0.5,0.5,-0.5])*aR;
		recy=array([-0.5,-0.5,0.5,0.5])*aR;
		[recx,recy]=mrotate(recx,recy,-alpha);
		recx=recx+Ox;recy=recy+Oy;
		y0=aR*cos(alpha-gamma)/cos(gamma)+recy[0];
		x0=-aR*sin(alpha-gamma)/cos(gamma)+recx[0];
		k=1/tan(beta/2);
		b=y0-k*x0;
		xm1=b/(-tan(beta/2)-k);
		ym1=-tan(beta/2)*xm1;
		am=sqrt(xm1**2+ym1**2);
		am=am+aR*(T0/T1)**2*0.5;#decrease interval size due to the factor T0/T1		
		dtheta=(2*beta)/Ntheta;drho=-log(am)/Nrho;	
		thetas=linspace(-beta,beta,Ntheta+1);thetas=thetas[:-1];
		rhos=linspace(log(am),0,Nrho+1);rhos=rhos[:-1];
		[theta,rho]=meshgrid(thetas,rhos);
		#t,x in lp
		tt1=(t**2-0.5)*aR;xx1=(x**2-0.5)*aR;
		[tt1,xx1]=mrotate(tt1,xx1,-alpha);
		tt1=tt1+Ox;xx1=xx1+Oy;
		thetat=arctan2(xx1,tt1);rhot=log(sqrt(tt1**2+xx1**2));
		emul=exp(reshape(rhot,shape(t)));
		thetat=thetat/2/beta;rhot=(rhot-log(am)/2)/(-log(am));thetat=ndarray.flatten(thetat);rhot=ndarray.flatten(rhot);
		#q,tau in lp
		thetaq=-arctan(q**2)+alpha;
		ttx=aR*(tau**2-0.5)*cos(alpha)+aR*sin(alpha)*(0.5)+Ox;
		tty=aR*(tau**2-0.5)*sin(alpha)-aR*cos(alpha)*(0.5)+Oy;
		rhoq=log((ttx+tty*tan(thetaq))*cos(thetaq));
		cosmul=cos(thetaq-alpha);
		thetaq=thetaq/2/beta;rhoq=(rhoq-log(am)/2)/(-log(am));thetaq=ndarray.flatten(thetaq);rhoq=ndarray.flatten(rhoq);
		#Jacobian hyperbolic integration, divided by 2x
		J=4*abs((aR**2*t)/(2*(Ox**2 + Oy**2) + aR**2*(1 - 2*t**2 + 2*t**4 - 2*x**2 + 2*x**4) + 2*aR*(Ox*(-1 + 2*t**2) + Oy*(-1 + 2*x**2))*cos(alpha) + 2*aR*(Ox - Oy + 2*Oy*t**2 - 2*Ox*x**2)*sin(alpha)));
		ker=lambda x: ((1/6.0*(2-abs(x))**3)*(abs(x)<2)*(abs(x)>=1)+(2/3.0-1/2.0*abs(x)**2*(2-abs(x)))*(abs(x)<1));#need [-N/2:N/2-1]
		B3theta=ker(theta[0,:]/2/beta*Ntheta);B3rho=ker((rho[:,0]-log(am)/2)/log(am)*Nrho);B3thrho=transpose(matrix(B3rho))*matrix(B3theta);
		fB3=array(fft.fftshift(fft.fft2(fft.ifftshift(B3thrho))));
		fZ=fzeta_loop_weights(Ntheta,Nrho,Ntheta*dtheta,Nrho*drho,0,8);
		fZB3=fft.fftshift(fZ/fB3/fB3);		

		fZgpu=fZB3[:,arange(0,Ntheta/2+1)];
		fZgpu=ndarray.flatten(transpose(array([real(ndarray.flatten(fZgpu)),ndarray.flatten(imag(fZgpu))])));


		#final normalization coefficient
		cc=X1*(t[0,1]-t[0,0])*(x[1,0]-x[0,0])/(theta[0,1]-theta[0,0])/(rho[1,0]-rho[0,0])/aR;
		
		#sorting
		ttn=thetat*Ntheta;xxn=rhot*Nrho;
		add=2;
		idthetatx=int32(floor(ttn))+Ntheta/2+add;idrhotx=int32(floor(xxn))+Nrho/2+add;
		thetae=zeros(size(theta[0,:])+2*add);rhoe=zeros(size(rho[:,0])+2*add);thetae[add:-add]=theta[0,:]/2/beta*Ntheta;rhoe[add:-add]=(rho[:,0]-log(am)/2)/(-log(am))*Nrho;
		dthetatx=ndarray.flatten(ttn)-thetae[idthetatx];
		drhotx=ndarray.flatten(xxn)-rhoe[idrhotx];
		idg=(idrhotx)*(Ntheta+2*add)+idthetatx;
		reorids=argsort(idg,0);		
		idg=idg[reorids];
		st=zeros((Nrho+2*add)*(Ntheta+2*add));nvals=zeros((Nrho+2*add)*(Ntheta+2*add),dtype=int32);
		k=0;
		while(k<Nt*Nx):
			j=1;
			st[idg[k]]=k;    		  
			if (k+j<Nt*Nx): ##to rewrite
				while(idg[k+j]==idg[k]):	
				        j=j+1;
					if (k+j>=Nt*Nx):
						break;
			nvals[idg[k]]=j;
			st[idg[k]+1:idg[min(k+j,Nt*Nx-1)]]=k+j;
			k=k+j;
		st[idg[Nt*Nx-1]+1:]=st[idg[Nt*Nx-1]]+j;#norm(st(idg+1)-st(idg)-nvals(idg))
		#fwd2
		ttn=thetaq*Ntheta;
		xxn=rhoq*Nrho;
		idthetatauq=int32(floor(ttn)+Ntheta/2+add);
		idrhotauq=int32(floor(xxn)+Nrho/2+add);

		thetae=zeros(size(theta[0,:])+2*add);rhoe=zeros(size(rho[:,0])+2*add);thetae[add:-add]=theta[0,:]/2/beta*Ntheta;rhoe[add:-add]=(rho[:,0]-log(am)/2)/(-log(am))*Nrho;
		dthetatauq=ndarray.flatten(ttn)-thetae[idthetatauq];
		drhotauq=ndarray.flatten(xxn)-rhoe[idrhotauq];
		
		###
		##sorting adj
		idgadj=idrhotauq*(Ntheta+2*add)+idthetatauq;
		reoridsadj=argsort(idgadj,0);
		idgadj=idgadj[reoridsadj];
		stadj=zeros((Nrho+2*add)*(Ntheta+2*add));nvalsadj=zeros((Nrho+2*add)*(Ntheta+2*add),dtype=int32);
		k=0;
		while(k<Nq*Ntau):
			j=1;
			stadj[idgadj[k]]=k;    		  			
			if (k+j<Nq*Ntau): ##to rewrite
				while(idgadj[k+j]==idgadj[k]):	
				        j=j+1;
					if (k+j>=Ntau*Nq):
						break;
			nvalsadj[idgadj[k]]=j;
			stadj[idgadj[k]+1:idgadj[min(k+j,Ntau*Nq-1)]]=k+j;
			k=k+j;
		stadj[idgadj[Ntau*Nq-1]+1:]=stadj[idgadj[Ntau*Nq-1]]+j;
		
		dthetatx=dthetatx[reorids];
		drhotx=drhotx[reorids];
		idthetatx=idthetatx[reorids];
		idrhotx=idrhotx[reorids];
	
		dthetatauq=dthetatauq[reoridsadj];
		drhotauq=drhotauq[reoridsadj];
		idthetatauq=idthetatauq[reoridsadj];
		idrhotauq=idrhotauq[reoridsadj];

		#to structure
		Nt=int32(Nt);Nx=int32(Nx);Nq=int32(Nq);Ntau=int32(Ntau);Ntheta=int32(Ntheta);Nrho=int32(Nrho);ni=int32(ni);
		st=int32(st);nvals=int32(nvals);reorids=int32(reorids);
		stadj=int32(stadj);nvalsadj=int32(nvalsadj);reoridsadj=int32(reoridsadj);
		idthetatx=int32(idthetatx);idrhotx=int32(idrhotx);
		idthetatauq=int32(idthetatauq);idrhotauq=int32(idrhotauq);

		t=float32(t);x=float32(x);q=float32(q);tau=float32(tau);theta=float32(theta);rho=float32(rho);

		alpha=float32(alpha);beta=float32(beta);Ox=float32(Ox);Oy=float32(Oy);aR=float32(aR);am=float32(am);
		J=float32(J);cc=float32(cc);
		emul=float32(emul);thetat=float32(thetat);rhot=float32(rhot);
		cosmul=float32(cosmul);thetaq=float32(thetaq);rhoq=float32(rhoq);
		dthetatx=float32(dthetatx);drhotx=float32(drhotx);
		dthetatauq=float32(dthetatauq);drhotauq=float32(drhotauq);

		emul=ndarray.flatten(emul);
		cosmul=ndarray.flatten(cosmul);J=ndarray.flatten(J);
		fZgpu=float32(fZgpu);

		self.lpRgpu0=lpRgpu(asarray([Nt,Nx,Nq,Ntau,Ntheta,Nrho,ni]),fZgpu,st,stadj,idthetatx,idrhotx,idthetatauq,idrhotauq,dthetatx,drhotx,dthetatauq,drhotauq,emul,cosmul,J,reorids,reoridsadj)
		self.cc=cc;
		self.Nt=Nt;
		self.Nx=Nx;
		self.Ntau=Ntau;
		self.Nq=Nq;

	def fwd(self,f):
		R=zeros([self.Ntau,self.Nq],dtype=float32);
		self.lpRgpu0.fwd(R,f)
		R=R*self.cc;
		return R
	
	def adj(self,R):
		f=zeros([self.Nx,self.Nt],dtype=float32);
		self.lpRgpu0.adj(f,R)
		f=f*self.cc;
		return f	


def fzeta_loop_weights(Ntheta,Nrho,betas,rhos,a,osthlarge):
	krho=arange(-Nrho/2,Nrho/2);
	Nthetalarge=osthlarge*Ntheta;
	thsplarge=arange(-Nthetalarge/2,Nthetalarge/2)/float32(Nthetalarge)*betas;
	fZ=array(zeros(shape=(Nrho,Nthetalarge)),dtype=complex);
	h=array(ones(Nthetalarge));
	# correcting=1+[-3 4 -1]/24;correcting(1)=2*(correcting(1)-0.5);
	# correcting=1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0;correcting[0]=2*(correcting[0]-0.5);
	correcting=1+array([-216254335,679543284,-1412947389,2415881496,-3103579086,2939942400,-2023224114,984515304,-321455811,63253516,-5675265])/958003200.0;correcting[0]=2*(correcting[0]-0.5);
	h[0]=h[0]*(correcting[0]);
	for j in range(1,size(correcting)):
		h[j]=h[j]*correcting[j];
		h[-1-j+1]=h[-1-j+1]*(correcting[j]);
	for j in range(0,size(krho)):
		fcosa=pow(cos(thsplarge),(-2*pi*1j*krho[j]/rhos-1-a));
		fZ[j,:]=fft.fftshift(fft.fft(fft.fftshift(h*fcosa)));
	fZ=fZ[:,range(Nthetalarge/2-Ntheta/2,Nthetalarge/2+Ntheta/2)];
	fZ=fZ*(thsplarge[1]-thsplarge[0]);
	#put imag to 0 for the border
	fZ[0]=0;#real(fZ(1,:));
	fZ[:,0]=0;#real(fZ(:,1));
	return fZ;

def mrotate(x1,x2,alpha):
		t1=x1*cos(alpha)+x2*sin(alpha);
		t2=-x1*sin(alpha)+x2*cos(alpha);
		return [t1,t2]



