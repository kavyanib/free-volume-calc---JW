import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import lfilter
import warnings
warnings.filterwarnings("ignore")
plt.style.use("classic")
from scipy.interpolate import interp1d
import sys
from scipy.special import gamma
from scipy.special import gammainc
import sys
for i2 in range(len(sys.argv)):
	if sys.argv[i2]=="-n":
		Na=float(sys.argv[i2+1])
	else:
		pass


def gamma_f(v,a,b):
	return a*(a*v)**(b-1)*np.exp(-a*v)/gamma(b)

#def f1(r,a1,b1,c1,d1,a2,a3,e1):
#	return a1*r**(-b1)-a2*r**(-c1)-a3*r**(-d1)-e1

def f2(r):
	return 4*(1/r**9-1/r**6)

def f4(r):
	return 4*(-12/r**13+6/r**7)

def f3(r,a1,a2,a3,a4,a5):
	return a1*np.exp(-a2*r)-a3/r**3-a4/r**4-a5/r**5

Nv=float(Na)
sigma=0.395
recp=0.00076*450+0.93452+(2*(0.06*450-2.45))/(Nv*14)
rho=0.5*43/recp
print(rho)
beta=760/(8.314*450)

filename2="table_result.txt"
writefile = open(filename2,'w')

filename3="pressure_vol.txt"
writefile2 = open(filename3,'w')

Ir3=[]
xr3=[]
filename="r3-2.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		xr3.append(float(matrix[0]))
		Ir3.append(float(matrix[1]))

Nay=[Na]#,20,30,40,50,60,70,80,90,100,150,200,250,300,350]
plt.figure()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.subplots_adjust(left=0.18,bottom=0.15)
c_str=["-b","-r","-g","-m","-c","-k","-y"]
count=0
for ik in range(len(Nay)):
	Na=Nay[ik]
	filename="g-"+str(Na)+"-450-py3.txt"
	readfile = open(filename,'r')
	sepfile = readfile.read().split('\n')
	readfile.close()
	gp0=[]
	rp0=[]
	for pair in sepfile:
		if pair == "":
			break
		elif pair[0] == "#":
			pass
		else:
			matrix = pair.split()
			rp0.append(float(matrix[0]))
			gp0.append(float(matrix[1]))
	#print(feff[0],reff[0])
	I0=np.zeros(len(rp0))
	rp02=[]
	I02=[]#np.zeros(len(rp02))
	I0_r=[]
	rp0_r=[]
	for i in range(len(rp0)):
		if gp0[i]>10**(-3):
			I0[i]=-gp0[i]*f4(rp0[i])*rp0[i]**3+Ir3[i]
	I02=[ii*0.76*0.395**2 for ii in I0]
	rp02=[ii*0.395 for ii in rp0]
	plt.plot(rp02,I02,'-b')
	plt.hlines(0,0,2.1)
	plt.ylabel(r"$-r^3g(r)\frac{d}{dr}u^{(0)}(r)~\mathrm{(kJ\cdot mole^{-1}\cdot nm^2)}$")
	plt.xlabel("$r~\mathrm{(nm)}$")
	plt.xlim(0,1)
	plt.ylim(-2,6)
	plt.text(0.25,0.2,"$r_c$")
	plt.text(0.35,4.2,"$r^+$")
	plt.text(0.44,0.2,"$r_{pm}$")
	plt.text(0.46,-0.8,"$r_{fm}$")
	plt.savefig("1.png",dpi=300)
	#plt.show()
	beta_c=0
	for i in range(len(rp0)-1):
		if rp0[i]>=2.07:
			break
		beta_c=beta_c+(I0[i]+I0[i+1])*(rp0[i+1]-rp0[i])/2
	beta_c=2*np.pi*rho*0.395**3*beta*beta_c/3+1
	print("compress",beta_c)
	rp01=[ii*0.395 for ii in rp0]
	I01=[ii*0.76*0.395**(2) for ii in I0]
	
	sum1=0
	sum2=0
	sum3=0
	sum4=0
	for i in range(len(rp01)):
		if rp01[i]<0.367 and rp01[i]>0:
			sum1=sum1+(I01[i+1]+I01[i])*(rp01[i+1]-rp01[i])*0.5
	print("sum1=",sum1)

	#0.0833434110843 0.0103059022294 0.0295259808684 0.00164552455028

	beta=1000/(8.314*450)
	recp=0.00076*450+0.93452+(2*(0.06*450-2.45))/(Nv*14)
	rho=0.5*43/recp
	vi=0.367**3*np.pi/6

	f=1/(1+(2/3)*np.pi*beta*rho*(sum1))
	print("probability",np.exp(-rho*(0.656*(0.367**3*np.pi/6))/f),f)
	#print("b=",beta_c,"a=",beta_c*7.51/f)
	
	writefile.write(str(np.exp(-rho*0.656*vi/f)))
	phi_plus=np.exp(-rho*0.656*vi/f)+0.06
	p_eff=8.314*450/(6.022*10**(-4)*f/rho)		
	writefile.write("\n")
	writefile2.write(str(p_eff*phi_plus*0.656*6.022*10**(-4)*vi+p_eff*6.022*10**(-4)*(1-phi_plus)*f/rho))
	writefile2.write("\n")


