import numpy as np
from abc import ABC, abstractmethod
import scipy
from scipy import linalg
import scipy.linalg

class BIBO(ABC):
    """[summary]:
        Abstract class for stability algorithms
    """
    @abstractmethod
    def conclusion(self):
        """[return :int]
        return:
            -1  : not BIBO
             0  : if cannot conclude about BIBO 
             1  : BIBO        
        """
        pass


class Gerschgorin(BIBO):
    """[summary]: this is sufficent condition for verifying BIBO attribute
    test_case: 
        A = [-3 -1 0] [2 -3 0] [-2 1 -4]  - > 1 
        A = [-2 -1 0] [4 -2 0] [-2 1 -1]  - > 0
    """
    def __init__(self,A: np.ndarray):
        self.A = A 
    def conclusion(self)-> int: 
        """[summary]
        """
        A = self.A
        diag = np.diag(A)
        B = np.sum(np.abs(A), axis=1)
        C = B - np.abs(diag) + diag
        result = np.all((C) <=0)
        
        if result:
            return 1 
        else: 
            return 0

class Lyapunov(BIBO):
    def __init__(self, A, P=None, Q=None):
        self.A = A
        shape = A.shape
        assert (P is None) ^ (Q is None), "please fill either P or Q"
        self. P = P
        self. Q = Q
        if P:
            assert P.shape == A.shape, "Invalid shape of matrix P"
        else:
            assert Q.shape == A.shape, "Invalid shape of matrix Q"
    @staticmethod
    def is_definite_positive_matrix(M):
        condition1 = np.all(np.linalg.eigvals(M) > 0)
        condition2 = np.allclose(M,M.T)
        return  condition1 and condition2

    def conclusion(self):
        if self.P:
            Q = -(self.P @ self.A + self.A.T @ self. P)
            result = Lyapunov.is_definite_positive_matrix(Q)
            
        else:
            P = linalg.solve_sylvester(a=self.A.T, b=self.A, q=-self.Q)
            result = Lyapunov.is_definite_positive_matrix(M=P)
        if result: 
            return 1
        else:
            return 0

class Hurwitz(BIBO):
    
    def __init__(self,A):
        self.A = A

    def conclusion(self,):
        result =  np.all(scipy.linalg.eigvals(self.A).real <0)
        if result:
            return 1 
        else: 
            return -1




if __name__ =='__main__':

    #A = np.array([-3, -1, 0, 2, -3, 0, -2, 1, -4]).reshape(3,3)
    A = np.array([-3,1,-1,-3]).reshape(2,2)
    Q = np.array([1,0,0,1]).reshape(2,2)
    something = Lyapunov(A, Q=Q)
    x = something.conclusion()
    print(x)