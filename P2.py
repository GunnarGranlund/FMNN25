import numpy as np

#===============================================================================
# The optimization problem class. The consctructor takes an objective function f,
# a grid x, the dimension n, and the opportunity to calculate the gradient as inputs
#===============================================================================

class optimizationProblem(object):
    def __init__(self, f, n):
        self.f = f
        self.h = 0.00001
        self.n = n
 
      
    def g(self, x):              # Calculates the gradient of the objective function f, for given 
        g = np.zeros(n)          # values of x1,..,xn. The vector e represents the indexes where the
        for i in range(n):       # stepsize, h should be added depending on what partial derivative we want
            e = np.zeros(n)      # to caclulate
            e[i] = self.h
            g[i] = (f(x + e) - f(x))/self.h
        return g

    def G(self, x):
        G = np.zeros((n, n))      # Calculates the hessian matrix of the objective function f for given
        for i in range(n):        # values of x1,...,xn. The vectors e1 and e2 represents the indexes where
            for j in range(n):    # the stepsize, h should be added depending on what partial derivative
                e1 = np.zeros(n)  # we want to calculate
                e2 = np.zeros(n)
                e1[i] = self.h
                e2[j] = self.h
                G[i,j] = (f(x + e1 + e2) - f(x + e1) - f(x + e2) + f(x))/self.h**2
                if i != j:        # Symmetrizing step
                    G[i, j] = G[j, i]
        return G
        
    
class BaseMethod(optimizationProblem):
    #Skicka in initial guess som input kanske
    #Behöver räkna ut vår Hessianmatris
    def __init__(self, initial):
        self.initial = initial
    


#Testsaker

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
 
x1 = 1.0
x2 = 1.0
x = np.append(x1, x2)
n = len(x)
opt = optimizationProblem(f, n)
print(opt.g(x), opt.G(x))

