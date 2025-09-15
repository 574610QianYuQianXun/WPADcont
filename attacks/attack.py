from abc import abstractproperty, abstractmethod

from utils.parameters import Params

class Attack:
    params: Params

    def __init__(self,params: Params):
        self.params = params

    @abstractmethod
    def perform_attack(self, *args, **kwargs) -> None:
        pass