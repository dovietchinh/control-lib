import numpy as np
from abc import ABC, abstractmethod
import scipy.linalg

class BIBO(ABC):
    """[summary]
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


class gerschgorin(BIBO):
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
        print(result)
        if result:
            return 1 
        else: 
            return 0

class lyapunov(BIBO):
    def __init__(self, A, P=None, Q=None):
        self.A = A
        shape = A.shape
        if 
        assert not (P ^ Q), "please fill either P or Q"
        self. P = P
        self. Q = Q
        if P:
            assert P.shape == A.shape, "Invalid shape of matrix P"
        else:
            assert Q.shape == A.shape, "Invalid shape of matrix Q"

    def is_definite_positive_matrix(cls,M):
        return 0
        

    def conclusion(self):
        if self.P:
            Q = -(self.P @ self.A + self.A.T @ self. P)
            result = lyapunov.is_definite_positive_matrix(Q)
            return result 
        else:
            P = scipy.linalg.solve_synvester(a=self.A.T, b=self.A, q=-self.Q)
            result = lyapunov.is_definite_positive_matrix(P)
            return result

if __name__ =='__main__':

    A = np.array([-3, -1, 0, 2, -3, 0, -2, 1, -4]).reshape(3,3)
    somethings = gerschgorin(A)
    somethings.conclusion()
    #print(somethings.conclusion())