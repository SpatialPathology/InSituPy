from pathlib import Path
from typing import List, Optional, Type

from .utils.utils import convert_to_list


class ModuleNotFoundOnWindows(ModuleNotFoundError):
    '''
    Code from https://github.com/theislab/scib/blob/main/scib/exceptions.py
    Information about structure: https://careerkarma.com/blog/python-super/

    Args:
        exception:
            Exception returned by OS.
    '''

    def __init__(self, exception):
        self.message = f"\n{exception.name} is not installed. " \
                       "This package could be problematic to install on Windows."
        super().__init__(self.message)

class InSituDataRepeatedCropError(Exception):
    """Exception raised if it is attempted to crop a
    InSituData object multiple times with the same cropping window.

    Args:
        xlim:
            Limits on x-axis.
        ylim:
            Limits on y-axis.
    """

    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        self.message = f"\nInSituData object has been cropped with the same limits before:\n" \
            f"xlim -> {xlim}\n" \
            f"ylim -> {ylim}"
        super().__init__(self.message)

class InSituDataMissingObject(Exception):
    """Exception raised if a certain object is not available in the InSituData object.
    Maybe it has to be read first

    Args:
        name:
            Name of object that is searched for.
    """

    def __init__(self, name):
        self.name = name
        self.message = f"\nInSituData object does not contain object `{name}`.\n" \
            f"Consider running `.read_{name}()` first."
        super().__init__(self.message)

class WrongNapariLayerTypeError(Exception):
    """Exception raised if current layer has not the right format.

    Args:
        found:
            Napari layer type found.
        wanted:
            Napari layer type wanted.
    """

    def __init__(self, found, wanted):
        self.message = f"\nNapari layer has wrong format ({found}) instead of {wanted}"
        super().__init__(self.message)

class NotOneElementError(Exception):
    """Exception raised if list contains not exactly one element.

    Args:
        list: List which does not contain one element.
    """

    def __init__(self, l):
        self.message = f"List was expected to contain one element but contained {len(l)}"
        super().__init__(self.message)

class UnknownOptionError(Exception):
    """Exception raised if a certain option is not found in a list of possible options.

    Args:
        name:
            Name of object that is searched for.
        available:
            List of available options.
    """

    def __init__(self, name, available):
        self.message = f"Option {name} is not available. Following parameters are allowed: {', '.join(available)}"
        super().__init__(self.message)


class NotEnoughFeatureMatchesError(Exception):
    """Exception raised if not enough feature matches were found.

    Args:
        number:
            Number of feature matches that were found.
        threshold:
            Threshold of number of feature matches.
    """

    def __init__(self,
                 number: str,
                 threshold: str
                 ):
        self.message = f"A maximum of {number} matched features were found. This was below the threshold of {threshold}."
        super().__init__(self.message)

class ModalityNotFoundError(Exception):
    """Exception raised if a certain modality is not found by InSituData read modules.

    Args:
        modality:
            Name of the modality (e.g. matrix)
    """

    def __init__(self,
                 modality: str
                 ):
        self.message = f"No `{modality}` modality found."
        super().__init__(self.message)

class InvalidFileTypeError(Exception):
    def __init__(self,
                 allowed_types: List[Type],
                 received_type: Type,
                 message: Optional[str] = None
                 ):
        # allowed_types = [allowed_types] if isinstance(allowed_types, str) else list(allowed_types)
        allowed_types = convert_to_list(allowed_types)
        allowed_types = [str(elem) for elem in allowed_types]
        received_type = str(received_type)
        if message is None:
            message = f"Invalid file type. Allowed file types: {', '.join(allowed_types)}. Received: {received_type}"
        self.message = message
        super().__init__(self.message)

class InvalidDataTypeError(Exception):
    def __init__(self,
                 allowed_types: List[Type],
                 received_type: Type,
                 message: Optional[str] = None
                 ):
        # allowed_types = [allowed_types] if isinstance(allowed_types, str) else list(allowed_types)
        allowed_types = convert_to_list(allowed_types)
        allowed_types = [str(elem) for elem in allowed_types]
        received_type = str(received_type)
        if message is None:
            message = f"Invalid data type. Allowed data types: {', '.join(allowed_types)}. Received: {received_type}"
        self.message = message
        super().__init__(self.message)

class InvalidXeniumDirectory(Exception):
    def __init__(self, directory):
        if (Path(directory) / ".ispy").exists():
            self.message = f"The directory '{directory}' does not contain the required 'experiment.xenium' file, but it contains an InSituPy project file. Try `InSituData.read()` instead."
        else:
            self.message = f"The directory '{directory}' does not contain the required 'experiment.xenium' file."
        super().__init__(self.message)