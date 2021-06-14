import numpy as np
from numpy.testing._private.utils import assert_array_max_ulp 
from scipy import integrate

class LinearSystem():
    
    def __init__(self,S=None,A=None,B=None,C=None,D=None, time_step=0.001):
        if S:
            assert isinstance(S,dict), 'Invalid data type of S'
            A = S.get('A')
            B = S.get('B')
            C = S.get('C')
            D = S.get('D')
        for i in [A,B,C,D]:
            assert isinstance(i,np.ndarray), f"Invalid data type of {i}"
            assert i.ndim==2, f'Invlid ndim of matrix {i}, {i}.ndim must be 2'
        assert A.shape[0] == A.shape[1] and A.shape[0] ==B.shape[0]  , f'Invalid shape of matrix A,B'
        self._A = A 
        self._B = B
        self._C = C 
        self._D = D
        self._states_shape = A.shape[0]
        self._inputs_shape = B.shape[1]
        self._outputs_shape = C.shape[0]
        self.time_step = 0.001 #second
        self._u = None 
        self._x0 = None
        




    @property
    def states_shape(self,) -> int:
        return self._statess_shape

    @property
    def inputs_shape(self,) -> int:
        return self._inputs_shape

    @property
    def outputs_shape(self,):
        return self._outputs_shape

    def get_dimension(self,) -> list:
        return self._states_shape, self._inputs_shape, self._outputs_shape
    
    def is_controlable(self) -> bool:
        pass 

    def is_observable(self,) -> bool:
        pass 

    def is_stable(self,) -> bool:
        pass 
    
    def set_input_function(self, function):
        self._u = function

    def initialize_statte(self, x0):
        assert type(x0) is np.ndarray, "Invalid type, x0 must be numpy.ndarray"
        x0.shape = (-1,1)
        assert x0.shape[0]==self._states_shape, f"Invalid dimension, system got the states_shape = {self._states_shape}"
        self._x0 = x0

    def simulink(self,time, time_step):
        assert self._x0, "please set initial state x0"
        assert self._u, "please set input before running simulink"
        function = 
        



"""[summary]

ABCD x0 , u 



"""

#def Routh(px):




if __name__ =='__main__':
    pass
    

        
