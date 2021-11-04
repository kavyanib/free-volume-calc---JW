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
	elif sys.argv[i2]=="-t1":
		t1=str(sys.argv[i2+1])
	else:
		pass

def omega_k(k,N):
	d=1.34
	E=np.sin(k*d)/(k*d)
	sum1=(1+(1-E**(N+1))/(1-E))#/float(N)
	return sum1

def lj_f(r):
	f=4*(-12*(1/r**13)+6*(1/r**7))
	return f

filename = "g-"+Na+".0-"+t1+"-py3.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
count = 0
dis = []
rdfa=[]
rdfa2=[]
for pair in sepfile:
	if pair == "":
		break
	elif pair[0]=="#" or pair[0]=="@":
		pass
	else:
		matrix=pair.split()
		dis.append(float(matrix[0]))
		rdfa.append(float(matrix[1])-1)
		rdfa2.append(float(matrix[1]))

dr=0.001
r=np.arange(dr,100,dr)
dk=np.pi/(dr*(len(r)+1))
k=np.linspace(dk,dk*len(r),len(r))

coeff_to_fourier=2.0*np.pi*r*dr
coeff_to_real=k*dk/(4*np.pi**2)

omega1k=np.zeros(len(r))
for i in range(len(r)):
	omega1k[i]=omega_k(k[i],float(Na))
omega1r=dst(coeff_to_real*omega1k,type=3)/r

omega_r_r=np.zeros(len(r))
omega_r=np.zeros(len(r))
rdfa_r=np.zeros(len(r))
for i in range(len(r)):
	omega_r_r[i]=r[i]**2*omega1r[i]
	omega_r[i]=omega1r[i]
	rdfa_r[i]=r[i]**2*rdfa[i]
#Fourier Transform 1D
first_k=dst(coeff_to_fourier*omega_r_r,type=2)/k
second_k=dst(coeff_to_fourier*omega_r,type=2)/k
third_k=dst(coeff_to_fourier*rdfa,type=2)/k
fourth_k=dst(coeff_to_fourier*rdfa_r,type=2)/k

inter_k=dst(coeff_to_fourier*rdfa2,type=2)/k

J1=np.zeros(len(r))
J2=np.zeros(len(r))
J3=np.zeros(len(r))
D=1/float(Na)
J4=np.zeros(len(r))
for i in range(len(r)):
	J1[i]=second_k[i]*fourth_k[i]#w(r') * |r'-r|^2
	J2[i]=first_k[i]*third_k[i]#r'^2w(r') 
	J3[i]=second_k[i]*third_k[i]#w(r')
	J4[i]=inter_k[i]
h2=dst(coeff_to_real*J1,type=3)/r
h3=dst(coeff_to_real*J2,type=3)/r
h4=dst(coeff_to_real*J3,type=3)/r
h5=dst(coeff_to_real*J4,type=3)/r
I=np.zeros(len(r))
for i in range(len(r)):
	if rdfa2[i]<10**(-3):
		pass
	else:
		I[i]=-2*0.5*(-h2[i]+h3[i]+r[i]**2*h4[i])*rdfa2[i]*lj_f(r[i])*r[i]**1/float(Na)

filename2 = "r3-2.txt"
writefile = open(filename2,'w')
for i in range(len(r)):
	writefile.write(str(r[i]))
	writefile.write("\t")
	writefile.write(str(I[i]))
	writefile.write("\n")

plt.figure()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.subplots_adjust(left=0.18,bottom=0.15)
plt.plot(r,I,'-b')
#plt.plot(k,second_k,'sr')
plt.xlim(0,3)
plt.xlabel("$r$")
plt.ylabel("$I(r)$")
plt.savefig("1.png",dpi=300)
#plt.ylim(-1,10)
#plt.show()

