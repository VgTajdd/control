from __future__ import division, print_function

import numpy as np
import scipy.linalg

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # Ref Bertsekas, p.151

    # First, try to solve the riccati equation.
    P = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # Compute the LQR gain.
    K = np.matrix(scipy.linalg.inv(R)*(B.T*P))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, P, eigVals

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # Ref Bertsekas, p.151

    # First, try to solve the riccati equation.
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # Compute the LQR gain.
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, P, eigVals