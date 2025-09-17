from typing import Callable

"""
Simple tuple class for constraints (and their Jacobians). May be expanded with autodiff... 
"""

class Constraint:
    
    def __init__(self, constraint: Callable, jacobian: Callable):
        self._constraint = constraint
        self._jacobian = jacobian
        self._assert_not_empty()
    
    @property
    def constraint(self):
        return self._constraint
    
    @property 
    def jacobian(self):
        return self._jacobian
    
    def _assert_not_empty(self):
        if self._constraint is None or self._jacobian is None:
            raise ValueError('Constraint or jacobian are empty')