#!/usr/bin/python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from lqr_methods import lqr

np.set_printoptions(precision=3)

# Define the parameters
m0 = 2.0
m1 = 2.5
m2 = 2.5
L1 = 5
L2 = 5
g  = 10

# Pendulum 2 Inertia
R1 = L1
I10 = m1*(L1*L1)/12
I1 = I10+m1*(R1*R1)
# Pendulum 2 Inertia
R2 = L2
I20 = m2*(L2*L2)/12
I2 = I20+m2*(R2*R2)

h1 = m0+m1+m2
h2 = m1*(L1/2)+m2*(L2/2)
h3 = m2*(L2/2)
h4 = m1*(L1/2)*(L1/2) + m2*L1*L2+I1
h5 = m2*(L2/2)*L1
h6 = m2*(L2/2)*(L2/2) + I2
h7 = m1*(L1/2)*g + m2*L1*g
h8 = m2*(L2/2)*g
a1 = (0.5*m1+m2)*L1
a2 = 0.5*m2*L2
a3 = ((1/3)*m1+m2)*L1*L1
a4 = 0.5*m2*L1*L2
a5 = (1/3)*m2*L2*L2
m = m0+m1+m2
f1 = (0.5*m1+m2)*g*L1
f2 = 0.5*m2*g*L2

# Define the system matrices
D0 = np.matrix([[m, a1, a2], [a1, a3, a4], [a2, a4, a5]])
Dg = np.matrix([[0, 0, 0], [0, -f1, 0], [0, 0, -f2]])
H = np.matrix([[1, 0, 0]]).T

iD0 = scipy.linalg.inv(D0)

# Define the State Space System
A_1 = np.zeros((3,3))
A_2 = np.eye(3)
A_3 = -iD0*Dg
A_4 = np.zeros((3,3))
A = np.concatenate((np.hstack((A_1, A_2)), np.hstack((A_3, A_4))))
B = np.concatenate((np.zeros((3,1)),iD0*H))
#C = np.eye(6);
#D = np.matrix([[0, 0, 0, 0, 0, 0]]).T;

eigVals, eigVecs = scipy.linalg.eig(A)
print("eig(A) = ", eigVals)

print("A = ", A)
print("B = ", B)

Q = np.diag([0.01,0.01,0.1,1,1,1])
R = np.diag([0.001])

K, P, eigVals = lqr(A,B,Q,R)

print("K = ", K)
print("eig(A-BK) = ", eigVals)

dt = 0.01
nsteps = 6000
total_time = nsteps*dt

time_array = np.linspace(0, nsteps*dt, nsteps, endpoint=True)
xk0 = [0, 0, 10*3.14/180, 0, 0, 0]
xk = np.matrix(xk0).T

X0 = []
X1 = []
X2 = []

firstTime = True
dxk = []

for t in time_array:
    if firstTime:
        firstTime = False
    else:
        xk += dxk*dt

    uk = -K*xk
    X0.append(xk[0,0])
    X1.append(xk[1,0]*(180.0/3.14))
    X2.append(xk[2,0]*(180.0/3.14))
    dxk = A*xk + B*uk

listofzeros = [0] * len(time_array)

plt.figure()
plt.subplot(311)
plt.plot(time_array, listofzeros, 'r--', time_array, X1)
plt.ylabel('angle 1')
plt.subplot(312)
plt.plot(time_array, listofzeros, 'r--', time_array, X2)
plt.ylabel('angle 2')
plt.subplot(313)
plt.plot(time_array, listofzeros, 'r--', time_array, X0)
plt.ylabel('position')
plt.xlabel('time(sec)')
plt.show()