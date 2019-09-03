#!/usr/bin/python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from lqr_methods import dlqr

'''
def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    return -K
'''
# Define the parameters
m0 = 100.5 ;
m1 =0.5 ;
m2 =0.75 ;
L1 =0.5 ;
L2 =0.75 ;
dt= 0.02 ;
g = 9.81;
'''
Q =np.diag([5 50 50 700 700 700]);
# One can also try the following
# Q =diag([700, 0, 0, 0, 0, 0])
# Q=diag([110 110 110 0 0 0])
R =1;
g=10;
# Define the system matrices
d0=[m0 + m1 + m2, (m1/2 + m2)*L1, (m2*L2)/2;(m1/2 + m2)*L1, (m1/3 + m2)*L1ˆ2,(m2*L1*L2)/2;(m2*L2)/2,(m2*L1*L2)/2,(m2*L2ˆ2)/3];
g=[0,0,0;0,-(m1/2 + m2)*L1*10,0;0,0,-(m2*L2*g)/2];
# Define the State Space System
a=[zeros(3),eye(3);-inv(d0)*dg,zeros(3)];
b=[zeros(3,1);inv(d0)*[1;0;0]];
c=eye(6);'''

# Pendulum 2 Inertia
R1 = L1;
I10 = m1*(L1*L1)/12;
I1 = I10+m1*(R1*R1);
# Pendulum 2 Inertia
R2 = L2;
I20 = m2*(L2*L2)/12;
I2 = I20+m2*(R2*R2);
h1 = m0+m1+m2;
h2 = m1*(L1/2)+m2*(L2/2);
h3 = m2*(L2/2);
h4 = m1*(L1/2)*(L1/2) + m2*L1*L2+I1;
h5 = m2*(L2/2)*L1;
h6 = m2*(L2/2)*(L2/2) + I2;
h7 = m1*(L1/2)*g + m2*L1*g;
h8 = m2*(L2/2)*g;
dt = 0.02 ;
a1 = (0.5*m1+m2)*L1;
a2 = 0.5*m2*L2;
a3 = ((1/3)*m1+m2)*L1*L1;
a4 = 0.5*m2*L1*L2;
a5 = (1/3)*m2*L2*L2;
m = m0+m1+m2;
f1 = (0.5*m1+m2)*g*L1;
f2 = 0.5*m2*g*L2;
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

print(A_1)
print(A_2)
print(A_3)
print(A_4)

A = np.concatenate((np.hstack((A_1, A_2)), np.hstack((A_3, A_4))))
B = np.concatenate((np.zeros((3,1)),iD0*H))
C = np.eye(6);
#D = [0; 0; 0; 0; 0; 0]; 

print(A) 
print(B)

#Q = np.matrix("1 0 0 0; 0 .0001 0 0 ; 0 0 1 0; 0 0 0 .0001")
#Q = np.diag([5,50,50,700,700,700])
Q = np.diag([1,1,1,1,1,1])
R = 1

K, X, eigVals = dlqr(A,B,Q,R)
print(K)
print(eigVals)
#print("double c[] = {%f, %f, %f, %f};" % (K[0,0], K[0,1], K[0,2], K[0,3]))

nsteps = 100
time = np.linspace(0, 2, nsteps, endpoint=True)
xk = np.matrix("0 ; 0 ; -0.02 ; 0;0;0")

X = []
T = []
U = []

for t in time:
    uk = -K*xk
    X.append(xk[0,0])
    #T.append(xk[1,0])
    #U.append(xk[2,0])
    #v = xk[1,0]
    #force = uk[0,0]
    #accel = force/(m0+m1+m2)
    #u = ((1-.404)*v + dt*accel)/.055/10
    #U.append(u)
    xk = A*xk + B*uk

plt.plot(time, X, label="cart position, meters")
#plt.plot(time, T, label='pendulum angle, radians')
#plt.plot(time, U, label='control voltage, decavolts')

plt.legend(loc='upper right')
plt.grid()
plt.show()