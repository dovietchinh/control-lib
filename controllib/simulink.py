import scipy.integrate
import scipy 
import numpy as np
import matplotlib.pyplot as plt

def function(t,x):
    A = 1
    return -x

def main():
    a  = scipy.integrate.solve_ivp(function, t_span=(0,10), y0 = np.array([10]), t_eval = np.linspace(0,10,100))      
    print(a)

if __name__ =='__main__':
    main()