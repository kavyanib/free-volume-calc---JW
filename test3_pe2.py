import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
plt.style.use("classic")

def f1(x,m,c):
	return m*x+c

def f2(x,m,c):
	return m*np.exp(x*c)
eta_T=[]
norm_fac=2.5*np.exp(-2000/(8.314*490))*np.exp(-2.9*4.18*1000/(8.314*490))
filename="eta_l_450k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1i=[]
d2i=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1i.append(float(matrix[0]))
		d2i.append(float(matrix[1])*np.exp(1458/450)*norm_fac)

filename="eta_l_460k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1i2=[]
d2i2=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1i2.append(float(matrix[0]))
		d2i2.append(float(matrix[1])*np.exp(1458/460)*norm_fac)

filename="eta_l_470k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1i1=[]
d2i1=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1i1.append(float(matrix[0]))
		d2i1.append(float(matrix[1])*np.exp(1458/470)*norm_fac)

filename="eta_l_480k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1j1=[]
d2j1=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1j1.append(float(matrix[0]))
		d2j1.append(float(matrix[1])*np.exp(1458/480)*norm_fac)

filename="eta_l_490k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1h=[]
d2h=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1h.append(float(matrix[0]))
		d2h.append(float(matrix[1])*np.exp(1458/490)*norm_fac)
eta_T.append(d2i[len(d2i)-1])
eta_T.append(d2i2[len(d2i2)-1])
eta_T.append(d2i1[len(d2i1)-1])
eta_T.append(d2j1[len(d2j1)-1])
eta_T.append(d2h[len(d2h)-1])
eta_T2=[]#1000
eta_T3=[]#900
eta_T4=[]#800
eta_T5=[]#700
eta_T6=[]#70
eta_T7=[]#500
eta_T2.append(d2i[len(d2i)-2])
eta_T2.append(d2i2[len(d2i2)-2])
eta_T2.append(d2i1[len(d2i1)-2])
eta_T2.append(d2j1[len(d2j1)-2])
eta_T2.append(d2h[len(d2h)-2])

eta_T3.append(d2i[len(d2i)-3])
eta_T3.append(d2i2[len(d2i2)-3])
eta_T3.append(d2i1[len(d2i1)-3])
eta_T3.append(d2j1[len(d2j1)-3])
eta_T3.append(d2h[len(d2h)-3])

eta_T4.append(d2i[len(d2i)-4])
eta_T4.append(d2i2[len(d2i2)-4])
eta_T4.append(d2i1[len(d2i1)-4])
eta_T4.append(d2j1[len(d2j1)-4])
eta_T4.append(d2h[len(d2h)-4])

eta_T5.append(d2i[len(d2i)-5])
eta_T5.append(d2i2[len(d2i2)-5])
eta_T5.append(d2i1[len(d2i1)-5])
eta_T5.append(d2j1[len(d2j1)-5])
eta_T5.append(d2h[len(d2h)-5])

eta_T6.append(d2i[5])
eta_T6.append(d2i2[5])
eta_T6.append(d2i1[5])
eta_T6.append(d2j1[5])
eta_T6.append(d2h[5])

eta_T7.append(d2i[len(d2i)-7])
eta_T7.append(d2i2[len(d2i2)-7])
eta_T7.append(d2i1[len(d2i1)-7])
eta_T7.append(d2j1[len(d2j1)-7])
eta_T7.append(d2h[len(d2h)-7])

eta_T8=[]
eta_T8.append(d2i[2])
eta_T8.append(d2i2[2])
eta_T8.append(d2i1[2])
eta_T8.append(d2j1[2])
eta_T8.append(d2h[2])


T=[450,460,470,480,490]
plt.figure()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.subplots_adjust(left=0.18,bottom=0.15)
plt.plot(d1i,d2i,'sb',label="$\mathrm{450~K}$")
plt.plot(d1i2,d2i2,'<g',label="$\mathrm{460~K}$")
plt.plot(d1i1,d2i1,'>m',label="$\mathrm{470~K}$")
plt.plot(d1j1,d2j1,'vc',label="$\mathrm{480~K}$")
plt.plot(d1h,d2h,'or',label="$\mathrm{490~K}$")

N_a=np.zeros(len(d1i))
for i in range(len(d1i)):
	N_a[i]=d1i[i]/28
Nc=300
n_f=np.arange(200,3500+20,20)
n_f_log=[]
y_f_log=[]
for i in range(len(N_a)):
	if N_a[i]>=Nc:
		n_f_log.append(np.log10(N_a[i]))
		y_f_log.append(np.log10(d2i[i]))
popt,pcov=curve_fit(f1,n_f_log,y_f_log)
print(popt,np.sqrt(np.diag(pcov)))
y_f=[]
for i in range(len(n_f)):
	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
n_f2=[ii*28 for ii in n_f]
plt.plot(n_f2,y_f,'-b')

n_f=np.arange(20,300+20,20)
n_f_log=[]
y_f_log=[]
for i in range(len(N_a)):
	if N_a[i]<=Nc:
		n_f_log.append(np.log10(N_a[i]))
		y_f_log.append(np.log10(d2i[i]))
popt,pcov=curve_fit(f1,n_f_log,y_f_log)
print(popt)
y_f=[]
for i in range(len(n_f)):
	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
n_f2=[ii*28 for ii in n_f]
plt.plot(n_f2,y_f,'-b')
N_a=np.zeros(len(d1h))
for i in range(len(d1h)):
	N_a[i]=d1h[i]/28
n_f=np.arange(200,3500+20,20)
n_f_log=[]
y_f_log=[]
for i in range(len(N_a)):
	if N_a[i]>=Nc:
		n_f_log.append(np.log10(N_a[i]))
		y_f_log.append(np.log10(d2h[i]))
popt,pcov=curve_fit(f1,n_f_log,y_f_log)
print(popt,np.sqrt(pcov[0][0]))
y_f=[]
for i in range(len(n_f)):
	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
n_f2=[ii*28 for ii in n_f]
plt.plot(n_f2,y_f,'-r')

n_f=np.arange(20,300+20,20)
n_f_log=[]
y_f_log=[]
for i in range(len(N_a)):
	if N_a[i]<=Nc:
		n_f_log.append(np.log10(N_a[i]))
		y_f_log.append(np.log10(d2h[i]))
popt,pcov=curve_fit(f1,n_f_log,y_f_log)
print(popt,pcov)
y_f=[]
for i in range(len(n_f)):
	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
n_f2=[ii*28 for ii in n_f]
plt.plot(n_f2,y_f,'-r')

filename="pe_dat_ham.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1h=[]
d2h=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1h.append(10**float(matrix[0]))
		d2h.append(10**float(matrix[1]))
plt.plot(d1h,d2h,'<',markerfacecolor='w',markeredgecolor='b')
filename="pearson_1994.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d14=[]
d24=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d14.append(10**float(matrix[0]))
		d24.append(10**float(matrix[1]))
plt.plot(d14,d24,'>',markerfacecolor='w',markeredgecolor='b')
filename="pearson_pe"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1=[]
d2=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1.append(10**float(matrix[0]))
		d2.append(10**float(matrix[1])*0.1)
plt.plot(d1,d2,'^',markerfacecolor='w',markeredgecolor='b')
#
#print(d2)
filename="padding_pe"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d11=[]
d12=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d11.append(10**float(matrix[0]))
		d12.append(10**float(matrix[1])*0.01*0.1)
plt.plot(d11,d12,'o',markerfacecolor='w',markeredgecolor='b')

filename="data_marina.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1=[]
d2=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1.append(float(matrix[0]))
		d2.append(float(matrix[1]))
plt.plot(d1,d2,'^',markerfacecolor='w',markeredgecolor='g')

filename="data_marina1.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
d1=[]
d2=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		d1.append(float(matrix[0]))
		d2.append(float(matrix[1]))
plt.plot(d1,d2,'d',markerfacecolor='w',markeredgecolor='g')

plt.legend(loc="upper left",fontsize=7,numpoints=1)


plt.text(4000,10**(-2),"$\mathrm{slope=1.54}$",color='r')
plt.text(12000,0.3,"$\mathrm{slope=3.77}$",color='r')

plt.text(200,20*10**(-2),"$\mathrm{slope=1.56}$",color='b')
plt.text(700,5*10,"$\mathrm{slope=3.87}$",color='b')


plt.xlabel("$M~\mathrm{(g\cdot mole^{-1})}$")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$\eta~\mathrm{(Pa\cdot s)}$")
#plt.xlim(5*10**2,5*10**5)
#plt.ylim(10**(0),10**6)
#plt.xlim(200,1*10**5)
#plt.ylim(10**(-3),1000)
plt.savefig("test3_pe.png",dpi=300)
plt.show()

plt.figure()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.subplots_adjust(left=0.18,bottom=0.15)
T1=[1000/ii for ii in T]
T11=np.arange(440,490,1)
T12=[1000/ii for ii in T11]
popt,pcov=curve_fit(f2,T1,eta_T)
popt2,pcov2=curve_fit(f2,T1,eta_T2)
popt3,pcov3=curve_fit(f2,T1,eta_T3)
popt4,pcov4=curve_fit(f2,T1,eta_T4)

popt5,pcov5=curve_fit(f2,T1,eta_T5)
popt6,pcov6=curve_fit(f2,T1,eta_T6)
popt7,pcov7=curve_fit(f2,T1,eta_T7)
popt8,pcov8=curve_fit(f2,T1,eta_T8)



print(popt)
xf=[]
yf=[]
yf2=[]
yf3=[]
yf4=[]

yf5=[]
yf6=[]
yf7=[]
yf8=[]
for i in range(len(T12)):
	xf.append(T12[i])
	yf.append(f2(T12[i],popt[0],popt[1]))
	yf2.append(f2(T12[i],popt2[0],popt2[1]))
	yf3.append(f2(T12[i],popt3[0],popt3[1]))
	yf4.append(f2(T12[i],popt4[0],popt4[1]))
	yf5.append(f2(T12[i],popt5[0],popt5[1]))
	yf6.append(f2(T12[i],popt6[0],popt6[1]))
	yf7.append(f2(T12[i],popt7[0],popt7[1]))
	yf8.append(f2(T12[i],popt8[0],popt8[1]))
#eta_T2=[]#1000
#eta_T3=[]#900
#eta_T4=[]#800
#eta_T5=[]#700
#eta_T6=[]#70
#eta_T7=[]#500
print("N=2000,",popt[1]*8.314/4.18)
print("N=1000,",popt2[1]*8.314/4.18)
print("N=900,",popt3[1]*8.314/4.18)
print("N=800,",popt4[1]*8.314/4.18)
print("N=700,",popt5[1]*8.314/4.18)
print("N=70,",popt6[1]*8.314/4.18)
print("N=500,",popt7[1]*8.314/4.18)
print("N=40,",popt8[1]*8.314/4.18)
plt.plot(xf,yf,"-b")
plt.plot(xf,yf2,"-b")
#plt.plot(xf,yf3,"-b")
#plt.plot(xf,yf4,"-b")
#plt.plot(xf,yf5,"-b")
plt.plot(xf,yf6,"-b")
plt.plot(xf,yf7,"-b")
plt.plot(xf,yf8,"-b")

plt.plot(T1,eta_T2,"^b")
#plt.plot(T1,eta_T3,"^b")
#plt.plot(T1,eta_T4,"sb")
#plt.plot(T1,eta_T5,"db")
plt.plot(T1,eta_T6,"^b")
plt.plot(T1,eta_T7,"^b")
plt.plot(T1,eta_T8,"^b")
plt.text(2.5,0.005,"$M=1.1~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.5,0.1,"$M=2~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.5,10,"$M=14~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.5,100,"$M=28~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.5,400,"$M=30~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.5,1000,"$M=60~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.5,3000,"$M=70~\mathrm{kg\cdot mole^{-1}}$")
plt.text(2.47,10000,"$M=139~\mathrm{kg\cdot mole^{-1}}$")
filename="data_70k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
T13=[]
eta13=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		T13.append(1000/(float(matrix[0])+273))
		eta13.append(float(matrix[1]))
#plt.plot(T13,eta13,"s",markerfacecolor='w',markeredgecolor='b')
plt.plot(T13,eta13,"s",markerfacecolor='w',markeredgecolor='b')
popt2,pcov2=curve_fit(f2,T13,eta13)

xf=[]
yf=[]
for i in range(len(T13)):
	xf.append(T13[i])
	yf.append(f2(T13[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-b')

filename="data_30k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
T12=[]
eta12=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		T12.append(1000/(float(matrix[0])+273))
		eta12.append(float(matrix[1]))
plt.plot(T12,eta12,"s",markerfacecolor='w',markeredgecolor='b')
popt2,pcov2=curve_fit(f2,T12,eta12)

xf=[]
yf=[]
for i in range(len(T12)):
	xf.append(T12[i])
	yf.append(f2(T12[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-b')

filename="data_60k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
T11=[]
eta11=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		T11.append(1000/(float(matrix[0])+273))
		eta11.append(float(matrix[1]))
plt.plot(T11,eta11,"s",markerfacecolor='w',markeredgecolor='b')

popt2,pcov2=curve_fit(f2,T11,eta11)

xf=[]
yf=[]
for i in range(len(T11)):
	xf.append(T11[i])
	yf.append(f2(T11[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-b')

filename="data_139k.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
T11=[]
eta11=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		T11.append(1000/(float(matrix[0])+273))
		eta11.append(float(matrix[1]))
plt.plot(T11,eta11,"s",markerfacecolor='w',markeredgecolor='b')

popt2,pcov2=curve_fit(f2,T11,eta11)

xf=[]
yf=[]
for i in range(len(T11)):
	xf.append(T11[i])
	yf.append(f2(T11[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-b')

filename="pearson_1000.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
dx=[]
dy=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		dx.append(float(matrix[0]))
		dy.append(10**float(matrix[1])*0.1)
plt.plot(dx,dy,"^",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
plt.plot(dx,dy2,'--b')
print("pearson1000=",popt[1]*8.314/4.18)
filename="pearson_500.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
dx=[]
dy=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		dx.append(float(matrix[0]))
		dy.append(10**float(matrix[1])*0.1)
plt.plot(dx,dy,"^",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
plt.plot(dx,dy2,'--b')
print("pearson500=",popt[1]*8.314/4.18)
print("dy=",dy)

filename="pearson_70.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
dx=[]
dy=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		dx.append(float(matrix[0]))
		dy.append(10**float(matrix[1])*0.1)
plt.plot(dx,dy,"^",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
plt.plot(dx,dy2,'--b')
print("pearson70=",popt[1]*8.314/4.18)
print("dy=",dy)

filename="pearson_37.txt"
readfile = open(filename,'r')
sepfile = readfile.read().split('\n')
readfile.close()
dx=[]
dy=[]
for pair in sepfile:
	if pair == "":
		break
	else:
		matrix = pair.split()
		dx.append(float(matrix[0]))
		dy.append(10**float(matrix[1])*0.1)
plt.plot(dx,dy,"^",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
plt.plot(dx,dy2,'--b')
print("pearson37=",popt[1]*8.314/4.18)
print("dy=",dy)
plt.plot(T1,eta_T,"^b")
plt.ylabel("$\eta~(\mathrm{Pa\cdot s})$")
plt.xlabel("$1000/T~\mathrm{(K^{-1})}$")
plt.yscale("log")
plt.xlim(1.8,3.2)
plt.ylim(10**(-3),5*10**4)
plt.savefig("test31.png",dpi=300)
plt.show()

'''
Ea_app=[]
M=[2000*28,1000*28,900*28,800*28]
Ea_app.append(popt[1]*8.314/4.18)
Ea_app.append(popt2[1]*8.314/4.18)
Ea_app.append(popt3[1]*8.314/4.18)
Ea_app.append(popt4[1]*8.314/4.18)
plt.figure()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.subplots_adjust(left=0.18,bottom=0.15)
plt.plot(M,Ea_app,'sb')
plt.xscale("log")
plt.show()
'''

