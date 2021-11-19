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

#### rotation - Linear transformation
avg_rotation = -1.9359664073211542 # In degree
avg_rotation_in_rad = avg_rotation*np.pi/180 # In radian
transform_matrix = np.array([[np.cos(avg_rotation_in_rad), -np.sin(avg_rotation_in_rad)], 
                             [np.sin(avg_rotation_in_rad), np.cos(avg_rotation_in_rad)]])
# transform matrix is:
#  -            -  -   -
# |  cos   -sin | |  x  |
# |  sin    cos | |  y  |
#  -            -  -   -
# New x is: x.cos - y.sin
# New y is: x.sin + y.cos 
#
# Now, we just need to correct the value of d2i, d2i2, d2i1, d2j1 and d2h:

d1i = np.log10(d1i)
d2i = np.log10(d2i)
d1i = np.array(d1i)*transform_matrix[0, 0] + np.array(d2i)*transform_matrix[0, 1]
d2i = np.array(d1i)*transform_matrix[1, 0] + np.array(d2i)*transform_matrix[1, 1] + 0.30
d1i = 10**d1i
d2i = 10**d2i


d1i2 = np.log10(d1i2)
d2i2 = np.log10(d2i2)
d1i2 = np.array(d1i2)*transform_matrix[0, 0] + np.array(d2i2)*transform_matrix[0, 1]
d2i2 = np.array(d1i2)*transform_matrix[1, 0] + np.array(d2i2)*transform_matrix[1, 1] + 0.30
d1i2 = 10**d1i2
d2i2 = 10**d2i2

d1i1 = np.log10(d1i1)
d2i1 = np.log10(d2i1)
d1i1 = np.array(d1i1)*transform_matrix[0, 0] + np.array(d2i1)*transform_matrix[0, 1]
d2i1 = np.array(d1i1)*transform_matrix[1, 0] + np.array(d2i1)*transform_matrix[1, 1] + 0.30
d1i1 = 10**d1i1
d2i1 = 10**d2i1

d1j1 = np.log10(d1j1)
d2j1 = np.log10(d2j1)
d1j1 = np.array(d1j1)*transform_matrix[0, 0] + np.array(d2j1)*transform_matrix[0, 1]
d2j1 = np.array(d1j1)*transform_matrix[1, 0] + np.array(d2j1)*transform_matrix[1, 1] + 0.30
d1j1 = 10**d1j1
d2j1 = 10**d2j1

d1h = np.log10(d1h)
d2h = np.log10(d2h)
d1h = np.array(d1h)*transform_matrix[0, 0] + np.array(d2h)*transform_matrix[0, 1]
d2h = np.array(d1h)*transform_matrix[1, 0] + np.array(d2h)*transform_matrix[1, 1] + 0.30
d1h = 10**d1h
d2h = 10**d2h


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



##############################################
# Printing the free volume theory results.
# The index "0:9" is for enentangled regime
# The indexes are:
# [560.0, 840.0, 1120.0, 1400.0, 1680.0, 1960.0, 2240.0, 2520.0, 2800.0, 5600.0, 
# 				8400.0, 11200.0, 14000.0, 16800.0, 19600.0, 22400.0, 25200.0, 28000.0, 56000.0]

T=[450,460,470,480,490]
plt.figure()
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.subplots_adjust(left=0.18,bottom=0.15)
# print("\nLen is: ", str(len(d1i)))
# print()
# print (d1i)
# print()

plt.plot(d1i[12::],d2i[12::],'sb',label="$\mathrm{450~K}$")
plt.plot(d1i2[12::],d2i2[12::],'<g',label="$\mathrm{460~K}$")
plt.plot(d1i1[12::],d2i1[12::],'>m',label="$\mathrm{470~K}$")
plt.plot(d1j1[12::],d2j1[12::],'vc',label="$\mathrm{480~K}$")
plt.plot(d1h[12::],d2h[12::],'or',label="$\mathrm{490~K}$")

## All of the d1i=d1i2=d1i1=d1j1=d1h. Why so many variables!?!

N_a=np.zeros(len(d1i))
for i in range(len(d1i)):
	N_a[i]=d1i[i]/28
Nc=300
n_f=np.arange(200,3500+20,20)

#####################################################
# This part is for the first slope, which is for the "3.87". 
# 
n_f_log=[]
y_f_log=[]
for i in range(len(N_a)):
	if N_a[i]>=Nc:
		n_f_log.append(np.log10(N_a[i]))
		y_f_log.append(np.log10(d2i[i]))
popt,pcov=curve_fit(f1,n_f_log,y_f_log)
print("\nThe 1st slope is: "+str(popt[0]))
first_slope = popt[0]

y_f=[]
for i in range(len(n_f)):
	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
n_f2=[ii*28 for ii in n_f]
plt.plot(n_f2,y_f,'-b')

#####################################################
# This part is for the second slope, which is for the "1.56". WE DO NOT NEED IT.
# Commented by Sajjad

# n_f=np.arange(20,300+20,20)
# n_f_log=[]
# y_f_log=[]
# for i in range(len(N_a)):
# 	if N_a[i]<=Nc:
# 		n_f_log.append(np.log10(N_a[i]))
# 		y_f_log.append(np.log10(d2i[i]))
# popt,pcov=curve_fit(f1,n_f_log,y_f_log)

# print("\nThe 2nd slope is: "+str(popt[0]))


#####################################################
# This part is for the third slope, which is for the "3.77". 
# 
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

y_f=[]
for i in range(len(n_f)):
	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
n_f2=[ii*28 for ii in n_f]
plt.plot(n_f2,y_f,'-r')

print("\nThe 3rd slope is: "+str(popt[0]))
third_slope = popt[0]

#####################################################
# This part is for the forth slope, which is for the "1.54". WE DO NOT NEED IT.
# Commented by Sajjad

# y_f=[]
# for i in range(len(n_f)):
# 	y_f.append(10**f1(np.log10(n_f[i]),popt[0],popt[1]))
# n_f2=[ii*28 for ii in n_f]
# plt.plot(n_f2,y_f,'-r')

# n_f=np.arange(20,300+20,20)
# n_f_log=[]
# y_f_log=[]
# for i in range(len(N_a)):
# 	if N_a[i]<=Nc:
# 		n_f_log.append(np.log10(N_a[i]))
# 		y_f_log.append(np.log10(d2h[i]))
# popt,pcov=curve_fit(f1,n_f_log,y_f_log)
# print("\nThe 4th slope is: "+str(popt[0])+"\n\n")



#############################################

#########################################
# Plotting Data by Harmandaris et al.

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

#########################################
# Plotting Data by Pearson 1994

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

#########################################
# Plotting Data by Pearson 1987

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

#########################################
# Plotting Data by Padding and Briels

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

#########################################
# Plotting Data by Najm and Savvas (part 1)

# filename="data_marina.txt"
# readfile = open(filename,'r')
# sepfile = readfile.read().split('\n')
# readfile.close()
# d1=[]
# d2=[]
# for pair in sepfile:
# 	if pair == "":
# 		break
# 	else:
# 		matrix = pair.split()
# 		d1.append(float(matrix[0]))
# 		d2.append(float(matrix[1]))
# plt.plot(d1,d2,'^',markerfacecolor='w',markeredgecolor='g')


#########################################
# Plotting Data by Najm and Savvas (part 2)

# filename="data_marina1.txt"
# readfile = open(filename,'r')
# sepfile = readfile.read().split('\n')
# readfile.close()
# d1=[]
# d2=[]
# for pair in sepfile:
# 	if pair == "":
# 		break
# 	else:
# 		matrix = pair.split()
# 		d1.append(float(matrix[0]))
# 		d2.append(float(matrix[1]))
# plt.plot(d1,d2,'d',markerfacecolor='w',markeredgecolor='g')


#######################################
# Plotting Data by Najm and Savvas - Corrected version (by Sajjad)

filename="data_marina_493K_sk.txt"
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
plt.plot(d1,d2,'o',markerfacecolor='w',markeredgecolor='g')


## pt. 2
filename="data_marina_483K_sk.txt"
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
plt.plot(d1,d2,'v',markerfacecolor='w',markeredgecolor='g')

## pt.3
filename="data_marina_473K_sk.txt"
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
plt.plot(d1,d2,'>',markerfacecolor='w',markeredgecolor='g')

## pt. 4
filename="data_marina_463K_sk.txt"
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
plt.plot(d1,d2,'<',markerfacecolor='w',markeredgecolor='g')

## pt. 5
filename="data_marina_453K_sk.txt"
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
plt.plot(d1,d2,'p',markerfacecolor='w',markeredgecolor='g')

## pt. 6
filename="data_marina_443K_sk.txt"
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
plt.plot(d1,d2,'P',markerfacecolor='w',markeredgecolor='g')

## pt. 7
filename="data_marina_433K_sk.txt"
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
plt.plot(d1,d2,'*',markerfacecolor='w',markeredgecolor='g')

## pt. 8
filename="data_marina_423K_sk.txt"
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
plt.plot(d1,d2,'D',markerfacecolor='w',markeredgecolor='g')


plt.legend(loc="upper left",fontsize=7,numpoints=1)


# plt.text(4000,10**(-2),"$\mathrm{slope=1.54}$",color='r')
plt.text(12000,0.3,"$\mathrm{slope=3.55}$",color='r')

# plt.text(200,20*10**(-2),"$\mathrm{slope=1.56}$",color='b')
inp_str = ""
plt.text(700,5*10,"$\mathrm{slope=3.64}$",color='b')


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



######################
##### From this line, it starts to generate plot $\eta$ vs. 1000/T.


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

print("\n\n>>>> END <<<<\n\n")

# Ea_app=[]
# M=[2000*28,1000*28,900*28,800*28]
# Ea_app.append(popt[1]*8.314/4.18)
# Ea_app.append(popt2[1]*8.314/4.18)
# Ea_app.append(popt3[1]*8.314/4.18)
# Ea_app.append(popt4[1]*8.314/4.18)
# plt.figure()
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)
# plt.gcf().set_size_inches(4,3,forward=True)
# plt.subplots_adjust(left=0.18,bottom=0.15)
# plt.plot(M,Ea_app,'sb')
# plt.xscale("log")
# plt.show()

