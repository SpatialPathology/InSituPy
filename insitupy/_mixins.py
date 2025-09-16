from copy import deepcopy
from dataclasses import fields


class DeepCopyMixin:
    def copy(self):
        '''
        Function to generate a deep copy of the current object.
        '''
        return deepcopy(self)

class GetMixin:
    def get(self, key):
        '''
        Function to retrieve and return an attribute of the current object.
        '''
        return getattr(self, key)

    def __getitem__(self, key):
        '''
        Function to retrieve and return an attribute of the current object.
        '''
        return getattr(self, key)


class _UpdatablePlottingConfig:
    def update_values(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}.")

    def show_all(self):
        print(f"Configuration parameters for {self.__class__.__name__}:")
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            print(f"\t{name}: {value}")
