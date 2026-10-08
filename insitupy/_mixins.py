import logging
from copy import deepcopy
from dataclasses import fields

logger = logging.getLogger(__name__)


class DeepCopyMixin:
    """Mixin that adds a :meth:`copy` method returning a deep copy of the object."""
    def copy(self):
        '''
        Function to generate a deep copy of the current object.
        '''
        return deepcopy(self)

class GetMixin:
    """Mixin that exposes attribute access via :meth:`get` and ``[]`` subscription."""
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
        """Update configuration attributes by name.

        Args:
            **kwargs: Keyword arguments mapping attribute names to new values.

        Raises:
            AttributeError: If a key is not an existing attribute of this object.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of {self.__class__.__name__}.")

    def show_all(self):
        """Log all dataclass field names and their current values at INFO level."""
        logger.info(f"Configuration parameters for {self.__class__.__name__}:")
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            logger.info(f"\t{name}: {value}")
