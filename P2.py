import numpy as np

#===============================================================================
# The optimization problem class. The consctructor takes an objective function f,
# a grid x, the dimension n, and the opportunity to calculate the gradient as inputs
#===============================================================================
# Vi borde skriva om detta så att function går från Rn -> R, enligt slidsen. Men har 
# mest testat så att jag får ut rätt saker nu i R

class optimizationProblem(object):
    def __init__(self, f, n):
        self.f = f
        self.h = 0.00001
        self.n = n
          
    def gradient(self, x):
        gradient = np.zeros(self.n)
        for i in range(self.n):
            x_new = x.copy()
            x_new[i] = x[i] + self.h
            gradient[i] = (f(x_new) - f(x))/self.h
        return gradient

    def hessian(self, x):
        hessian = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                x_new1 = x.copy()
                x_new1[i] = x[i] + self.h
                x_new2 = x.copy()
                x_new2[i] = x[i] - self.h
                hessian[i,j] = 
        return (f(x + self.h) - 2*f(x) + f(x - self.h))/self.h**2
        
    
class BaseMethod(optimizationProblem):
    #Skicka in initial guess som input kanske
    #Behöver räkna ut vår Hessianmatris
    def __init__(self, initial):
        self.initial = initial
    


#Testsaker

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
 
x1 = np.linspace(0.0, 9.0, 10)
x2 = np.linspace(0.0, 9.0, 10)

x = np.array(([x1, x2]))
xx1 = 1.0
xx2 = 1.0

xx = np.append(xx1, xx2)
#xx = np.append(x1.T, x2.T)
opt = optimizationProblem(f, 2)
print(opt.gradient(xx))

