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
######
## 2.9 is "Delta E" in k cal
# 4.18*1000 is the conversion factor to change "k cal" to "joules"
delta_E_in_joule =  np.exp(-2.9*4.18*1000/(8.314*490))
norm_fac=2.5*np.exp(-2000/(8.314*490))*delta_E_in_joule
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

## Transform the di1 etc to "log" scale, transform them, and then take them back:
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
## Re-scaling and transformation is done!


## after transformation:
# eta_T is for Mw = 72000
# eta_T2 is for Mw = 30000
# eta_T3 is for Mw = 28000
# eta_T4 is for Mw = 24000
# eta_T5 is for Mw = 20000
# eta_T6 is for Mw = 17000
# eta_T7 is for Mw = 14000
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

eta_T6.append(d2i[len(d2i)-6])
eta_T6.append(d2i2[len(d2i2)-6])
eta_T6.append(d2i1[len(d2i1)-6])
eta_T6.append(d2j1[len(d2j1)-6])
eta_T6.append(d2h[len(d2h)-5])

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

filename="pe_dat_ham_Mod.txt"
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

filename="pearson_1994_Mod.txt"
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

filename="pearson_pe_Mod"
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

filename="padding_pe_Mod"
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
plt.text(1700,5*10,"$\mathrm{slope=3.64}$",color='b')


plt.xlabel("$M_w~\mathrm{(g\cdot mole^{-1})}$")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$\eta~\mathrm{(Pa\cdot s)}$")
#plt.xlim(5*10**2,5*10**5)
#plt.ylim(10**(0),10**6)
#plt.xlim(200,1*10**5)
#plt.ylim(10**(-3),1000)
plt.savefig("eta-vs-Mw.png",dpi=300)
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
popt,pcov=curve_fit(f2,T1,eta_T)     # eta_T is for Mw = 72000 (based on corrected data)
popt2,pcov2=curve_fit(f2,T1,eta_T2)  # eta_T2 is for Mw = 30000 (based on corrected data)
popt3,pcov3=curve_fit(f2,T1,eta_T3)  # eta_T3 is for Mw = 28000 (based on corrected data)
popt4,pcov4=curve_fit(f2,T1,eta_T4)  # eta_T4 is for Mw = 24000 (based on corrected data)

popt5,pcov5=curve_fit(f2,T1,eta_T5)  # eta_T5 is for Mw = 20000 (based on corrected data)
popt6,pcov6=curve_fit(f2,T1,eta_T6)  # eta_T6 is for Mw = 17000 (based on corrected data)
popt7,pcov7=curve_fit(f2,T1,eta_T7)  # eta_T7 is for Mw = 14000 (based on corrected data)
popt8,pcov8=curve_fit(f2,T1,eta_T8)  # eta_T8 is for Mw = 72000 (based on corrected data)



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

plt.plot(xf,yf,"-r") # is for Mw = 70000
plt.plot(xf,yf2,"-r") # is for Mw = 30000
plt.plot(xf,yf3,"-r") # Mw = 28000
# plt.plot(xf,yf4,"-r")
plt.plot(xf,yf5,"-r")
# plt.plot(xf,yf6,"-b") # is for Mw = 17000
plt.plot(xf,yf7,"-r") # Is for Mw = 14000
# plt.plot(xf,yf8,"-r")

plt.plot(T1,eta_T,"^r", zorder = 10) # is for Mw = 70000
plt.plot(T1,eta_T2,"^r") # is for Mw = 30000
plt.plot(T1,eta_T3,"^r") # Mw = 28000
# plt.plot(T1,eta_T4,"^r")
plt.plot(T1,eta_T5,"^r")
# plt.plot(T1,eta_T6,"^b") # is for Mw = 17000
plt.plot(T1,eta_T7,"^r") # is for Mw = 14000
# plt.plot(T1,eta_T8,"^r")

# plt.text(2.5,0.005,"$M=1.1~\mathrm{k}$", fontsize=8, color = 'b')
# plt.text(2.5,0.1,"$M=2~\mathrm{k}$", fontsize=8, color = 'b')
plt.text(2.5,3.5,"$M=14~\mathrm{k}$", fontsize=7.5, color = 'r')
plt.text(2.75,3.5,",", fontsize=8)
plt.text(2.8,3.5,"$M=11\mathrm{k}$", fontsize=7.5, color = 'b')
plt.text(2.5,11,"$M=20~\mathrm{k}$", fontsize=7.5, color = 'r')
plt.text(2.5,22,"$M=17~\mathrm{k}$", fontsize=7.5, color = 'g')
plt.text(2.5,35,"$M=28~\mathrm{k}$", fontsize=7.5, color = 'r')
plt.text(2.5,55,"$M=30~\mathrm{k}$", fontsize=7.5, color = 'r')
plt.text(2.5,120,"$M=28~\mathrm{k}$", fontsize=7.5, color = 'b')
plt.text(2.5,250,"$M=35~\mathrm{k}$", fontsize=7.5, color = 'g')
plt.text(2.5,1000,"$M=60~\mathrm{k}$", fontsize=7.5, color = 'g')
plt.text(2.5,3000,"$M=70~\mathrm{k}$", fontsize=7.5, color = 'g')
plt.text(2.75,3000,",", fontsize=7.5)
plt.text(2.8,3000,"$M=70\mathrm{k}$", fontsize=7.5, color = 'r')
plt.text(2.47,14000,"$M=130~\mathrm{k}$", fontsize=7.5, color = 'g')


###### Collecting the slop of the eta-vs-1/T for the plot of Ea-vs-Mw:
## for the free volume theory:
eta_free_vol = np.array([popt[1], popt2[1], popt3[1], popt4[1], popt5[1], popt7[1]])
eta_free_vol = eta_free_vol * 8.314/4.18
eta_free_vol_Mw = np.array([71000, 30000, 28000, 24000, 20000, 14000])
eta_free_vol_Mw = eta_free_vol_Mw/1000
#########################
## Plotting the data by Najm abd Savvas: (corrected) by Sajjad

## pt. 1
filename="data_marina_Mw_17K_sk.txt"
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
		T12.append(1000/(float(matrix[0])))
		eta12.append(float(matrix[1]))
plt.plot(T12,eta12,"s",markerfacecolor='w',markeredgecolor='g')
popt2,pcov2=curve_fit(f2,T12,eta12)

xf=[]
yf=[]
for i in range(len(T12)):
	xf.append(T12[i])
	yf.append(f2(T12[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-g')
eta_Najm = np.array([popt[1]*8.314/4.18])
eta_Najm_mw = np.array([17000/1000])

## pt. 2
filename="data_marina_Mw_35K_sk.txt"
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
		T11.append(1000/(float(matrix[0])))
		eta11.append(float(matrix[1]))
plt.plot(T11,eta11,"s",markerfacecolor='w',markeredgecolor='g')

popt2,pcov2=curve_fit(f2,T11,eta11)

xf=[]
yf=[]
for i in range(len(T11)):
	xf.append(T11[i])
	yf.append(f2(T11[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-g')

eta_Najm = np.append(eta_Najm, popt2[1]*8.314/4.18)
eta_Najm_mw = np.append(eta_Najm_mw, 35000/1000)


## pt. 3
filename="data_marina_Mw_60K_sk.txt"
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
		T11.append(1000/(float(matrix[0])))
		eta11.append(float(matrix[1]))
plt.plot(T11,eta11,"s",markerfacecolor='w',markeredgecolor='g')

popt2,pcov2=curve_fit(f2,T11,eta11)

xf=[]
yf=[]
for i in range(len(T11)):
	xf.append(T11[i])
	yf.append(f2(T11[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-g')

eta_Najm = np.append(eta_Najm, popt2[1]*8.314/4.18)
eta_Najm_mw = np.append(eta_Najm_mw, 60000/1000)

## pt. 4
filename="data_marina_Mw_70K_sk.txt"
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
		T13.append(1000/(float(matrix[0])))
		eta13.append(float(matrix[1]))
#plt.plot(T13,eta13,"s",markerfacecolor='w',markeredgecolor='b')
plt.plot(T13,eta13,"s",markerfacecolor='w',markeredgecolor='g')
popt2,pcov2=curve_fit(f2,T13,eta13)

xf=[]
yf=[]
for i in range(len(T13)):
	xf.append(T13[i])
	yf.append(f2(T13[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-g')

eta_Najm = np.append(eta_Najm, popt2[1]*8.314/4.18)
eta_Najm_mw = np.append(eta_Najm_mw, 70000/1000)

## pt. 5
filename="data_marina_Mw_130K_sk.txt"
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
		T11.append(1000/(float(matrix[0])))
		eta11.append(float(matrix[1]))
plt.plot(T11,eta11,"s",markerfacecolor='w',markeredgecolor='g')

popt2,pcov2=curve_fit(f2,T11,eta11)

xf=[]
yf=[]
for i in range(len(T11)):
	xf.append(T11[i])
	yf.append(f2(T11[i],popt2[0],popt2[1]))
plt.plot(xf,yf,'-g')

eta_Najm = np.append(eta_Najm, popt2[1]*8.314/4.18)
eta_Najm_mw = np.append(eta_Najm_mw, 130000/1000)

#################################
### Plotting other experimintal results (read README.md file for more info)

## pt. 1
filename="pearson_1000.txt"  # This is for Mw = 28000 (sample 8 in paper)
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
plt.plot(dx,dy,"^c",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
plt.plot(dx,dy2,'-b')
print("pearson1000=",popt[1]*8.314/4.18)

eta_Pearson = np.array([popt[1]*8.314/4.18])
eta_Pearson_Mw = np.array([28000/1000])

## pt. 2
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

eta_Pearson = np.append(eta_Pearson, popt[1]*8.314/4.18)
eta_Pearson_Mw = np.append(eta_Pearson_Mw, 11000/1000)

# pt. 3
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
# plt.plot(dx,dy,"^",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
# plt.plot(dx,dy2,'--b')
print("pearson70=",popt[1]*8.314/4.18)
print("dy=",dy)

eta_Pearson = np.append(eta_Pearson, popt[1]*8.314/4.18)
eta_Pearson_Mw = np.append(eta_Pearson_Mw, 2000/1000)

## pt. 4
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
# plt.plot(dx,dy,"^",markeredgecolor='b',markerfacecolor='w')
popt,pcov=curve_fit(f2,dx,dy)
dy2=[]
for i in range(len(dx)):
	dy2.append(f2(dx[i],popt[0],popt[1]))
# plt.plot(dx,dy2,'--b')
print("pearson37=",popt[1]*8.314/4.18)
print("dy=",dy)

eta_Pearson = np.append(eta_Pearson, popt[1]*8.314/4.18)
eta_Pearson_Mw = np.append(eta_Pearson_Mw, 1100/1000)


########  End of collecting and preparing for plot ###########
##############################################################

#### Now plotting eta-vs-1/T :

plt.ylabel("$\eta~(\mathrm{Pa\cdot s})$")
plt.xlabel("$1000/T~\mathrm{(K^{-1})}$")
plt.yscale("log")
plt.xlim(1.8,3.2)
plt.ylim(10**(-0),1*10**5)
plt.savefig("eta-vs-1_over_T.png",dpi=300)
plt.show()

#############################

#### Now plotting Ea-vs-Mw:

def ea_mw(x, m, c):
	return -m + c*np.log(x) #m*x+c


plt.plot(eta_free_vol_Mw, eta_free_vol, '^r', label = 'Theory')
popt, _ = curve_fit(ea_mw, eta_free_vol_Mw, eta_free_vol)
m, c = popt
yFit_eta_free_vol = ea_mw(eta_free_vol_Mw, m, c)
plt.plot(eta_free_vol_Mw, yFit_eta_free_vol, '--r')
plt.text(40, 2.5,'-'+str(round(m, 4))+'+'+str(round(c, 4))+r'$\times$'+"$log M$", fontsize=8, color = 'r')


plt.plot(eta_Najm_mw, eta_Najm, 'sg', label = 'Experiment')
popt, _ = curve_fit(ea_mw, eta_Najm_mw, eta_Najm)
m, c = popt
yFit_eta_Najm = ea_mw(eta_Najm_mw, m, c)
plt.plot(eta_Najm_mw, yFit_eta_Najm, '--g')
plt.text(40, 8.5, '-'+str(round(m, 4))+'+'+str(round(c, 4))+r'$\times$'+"$log M$", fontsize=8, color = 'g')


# plt.plot(eta_Pearson_Mw, eta_Pearson, 'v-b', label = 'Pearson 1987')
plt.ylabel("$E_{a}~(\mathrm{kcal\cdot mol^{-1}})$")
plt.xlabel("$M_{w}~\mathrm{(kg\cdot mole^{-1})}$")
# plt.yscale("log")
# plt.xlim(1.8,3.2)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.gcf().set_size_inches(4,3,forward=True)
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend()
plt.ylim([0,20])
plt.savefig("Ea-vs-Mw.png",dpi=300)
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

