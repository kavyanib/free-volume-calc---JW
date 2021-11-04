import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
plt.style.use("classic")
from scipy.fftpack import dst
import sys
for i2 in range(len(sys.argv)):
	if sys.argv[i2]=="-n":
		Na=str(sys.argv[i2+1])
	else:
		pass

def lj_p(r):
	f=4*(1)*((1/(r)**12)-(1/(r)**6))
	return f
def omega(k,N):
	d=1.34
	E=np.sin(k*d)/(k*d)
	return (1-E**2-2*E/N+2*E**(N+1)/N)/(1-E)**2

sigma=1
temp1=[450]#[400,390,380,370,360,350]#[450,440,430,420,410]
Nv=float(Na)
for i5 in range(len(temp1)):
	beta=(760)/(temp1[i5]*8.314)
	recp=0.00076*temp1[i5]+0.93452+(2*(0.06*temp1[i5]-2.45))/(Nv*14)
	rho=0.5*43/recp
	rho=float(rho)*0.395**3
	dr=0.001
	r=np.arange(dr,100,dr)
	dk=np.pi/(dr*(len(r)+1))
	k=np.linspace(dk,dk*len(r),len(r))
	g=np.zeros(len(r))
	h=np.zeros(len(r))
	y=np.zeros(len(r))
	y_o=np.zeros(len(r))
	err=10
	count=0
	alpha=0.2
	d=0
	c=np.zeros(len(r))
	for i in range(len(r)):
		if r[i]<=d:
			pass
		else:
			g[i]=np.exp(-beta*lj_p(r[i]))
			h[i]=g[i]-1
			c[i]=h[i]
	coeff_to_fourier=2.0*np.pi*r*dr
	coeff_to_real=k*dk/(4*np.pi**2)

	hk=np.zeros(len(r))
	ck=np.zeros(len(r))
	ci=np.zeros(len(r))
	wk=np.zeros(len(r))
	err=1
	count=0
	#gk=dst(coeff_to_fourier*g,type=2)/k
	#gr=dst(coeff_to_real*gk,type=3)/r
	N=float(Na)
	#N=int(N)
	frac=0.8
	while(err>10**(-6)):
		#get direct correlation function to approximate c according to closure
		for i in range(len(r)):
			c[i]=(np.exp(-beta*lj_p(r[i]))-1)*(1+h[i]-c[i])
		#Fourier Transform 1D
		ck=dst(coeff_to_fourier*c,type=2)/k
		for i in range(len(h)):
			hk[i]=omega(k[i],N)**2*ck[i]/(1-rho*omega(k[i],N)*ck[i])#omega12[i]**2*ck[i]/(1-rho*omega12[i]*ck[i])
		#Inverse Fourier Transform 1D
		h2=dst(coeff_to_real*hk,type=3)/r
		err=0
		for i in range(len(h)):
			err=err+(h[i]-h2[i])**2/len(h)
			h[i]=frac*(h2[i])+(1-frac)*h[i]
		err=np.sqrt(err)
		if count>=1000:
			break
		count=count+1
		if count%1==0:
			print(count,err,temp1[i5])

	plt.figure()
	plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	plt.rc('text', usetex=True)
	plt.gcf().set_size_inches(4,3,forward=True)
	plt.subplots_adjust(left=0.18,bottom=0.15)
	g=[ii+1 for ii in h]
	plt.plot(r,g,'-b')
	#plt.plot(r,g,'o',markerfacecolor='w',markeredgecolor='b',markersize=4)
	plt.xlim(0,4)
	plt.xlabel("$r$")
	plt.ylabel("$g(r)$")
	plt.savefig("g-"+str(N)+"-"+str(temp1[i5])+".png",dpi=300)

	filename2 = "g-"+str(N)+"-"+str(temp1[i5])+"-py3.txt"
	writefile = open(filename2,'w')
	for i in range(len(r)):
		writefile.write(str(r[i]))
		writefile.write("\t")
		writefile.write(str(g[i]))
		writefile.write("\n")


