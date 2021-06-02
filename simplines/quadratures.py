import numpy as np


# ....
def gauss_legendre(ordergl,tol=10e-14):
    """
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
    """
    m = ordergl + 1
    from math import cos,pi
    from numpy import zeros

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p1,dp

    A = zeros(m)
    x = zeros(m)
    nRoots = (m + 1)// 2          # Number of non-neg. roots
    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30):
            p,dp = legendre(t,m)          # Newton-Raphson
            dt = -p/dp; t = t + dt        # method
            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    return x,A
# ...

def quad(f,a,b):
    n=1000
    x,w = gauss_legendre(ordergl=n)
    G = 0
    for i in range(n):
        G = G + w[i]*f(0.5*(b-a)*x[i]+ 0.5*(b+a))
    G = 0.5*(b-a)*G
    return G


""" test """

def exp(x):
    return np.exp(x)
g=quad(exp,-3,3)
print('G=',g)   # G= 20.0357487497486

