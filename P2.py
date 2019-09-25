import numpy as np

#===============================================================================
# The optimization problem class. The consctructor takes an objective function f,
# a grid x, and the opportunity to calculate the gradient as inputs
#===============================================================================
# Vi borde skriva om detta så att function går från Rn -> R, enligt slidsen. Men har 
# mest testat så att jag får ut rätt saker nu i R

class optimizationProblem(object):
    def __init__(self, f, x, gradient = True):
        self.f = f
        self.x = x
        self.h = 0.000001

        if gradient:
            self.gradient = np.zeros(len(x))
            for i in range(len(x)):
                self.gradient[i] = (f(x[i] + self.h) - f(x[i]))/self.h

    
class QuasiNewton(optimizationProblem):
    #Skicka in initial guess som input kanske
    #Behöver räkna ut vår Hessianmatris

#Testsaker

def f(x):
    return x**2 + 2*x + 1
    
x = np.linspace(0.0, 10.0, 10)
opt = optimizationProblem(f, x)

