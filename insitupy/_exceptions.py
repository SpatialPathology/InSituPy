from pathlib import Path

from ._constants import ISPY_METADATA_FILE
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

class NoImageOverlapError(Exception):
    """Raised when a crop region does not overlap with the image at all.

    Args:
        xlim: x-axis limits of the crop region.
        ylim: y-axis limits of the crop region.
    """

    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        self.message = (
            f"The crop region (xlim={xlim}, ylim={ylim}) does not overlap "
            f"with the image extent. No image data can be cropped."
        )
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
        partial_result:
            Best FeatureMatchResult found before failure (for QC output). May be None.
    """

    def __init__(self,
                 number: str,
                 threshold: str,
                 partial_result=None,
                 ):
        self.partial_result = partial_result
        self.message = f"A maximum of {number} matched features were found. This was below the threshold of {threshold}."
        super().__init__(self.message)

class ModalityNotFoundError(Exception):
    """Exception raised if a certain modality is not found by InSituData read modules.

    Args:
        modality:
            Name of the modality (e.g. table)
    """

    def __init__(self,
                 modality: str
                 ):
        self.message = f"No '{modality}' modality found."
        super().__init__(self.message)



class ModalityNotFoundWarning(UserWarning):
    """Warning raised if a certain modality is not found by InSituData read modules.

    Args:
        modality:
            Name of the modality (e.g. table)
    """
    def __init__(self, modality: str):
        message = f"No '{modality}' modality found."
        super().__init__(message)

class InvalidFileTypeError(Exception):
    """Raised when a file path has an extension that is not in the allowed set."""
    def __init__(self,
                 allowed_types: list[type],
                 received_type: type,
                 message: str | None = None
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
    """Raised when a data object has a type that is not in the allowed set."""
    def __init__(self,
                 allowed_types: list[type],
                 received_type: type,
                 message: str | None = None
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
    """Raised when a path is not a valid Xenium output directory."""
    def __init__(self, directory):
        directory = Path(directory)
        if (directory / ".ispy").exists():
            self.message = f"The directory '{directory}' does not contain the required 'experiment.xenium' file, but it contains an InSituPy project file. Try `InSituData.read()` instead."
        else:
            self.message = f"The directory '{directory}' is not a valid Xenium directory. It does not contain the required 'experiment.xenium' metadata file."
        super().__init__(self.message)


class InSituDataConstructorPathError(ValueError):
    """Raised when a saved InSituPy project path is passed to ``InSituData()``
    instead of ``InSituData.read()``."""
    def __init__(self, path):
        super().__init__(
            f"'{path}' is already a saved InSituPy project (contains "
            f"'{ISPY_METADATA_FILE}'). Load it with `InSituData.read(path)` "
            f"instead of `InSituData(path)`."
        )


class MissingPackageError(ImportError):
    """Raised when an optional dependency is required but not installed."""
    def __init__(self, package_name: str, installation_command: str | None):
        if installation_command is None:
            installation_command = f"pip install {package_name}"

        super().__init__(
            f"The package `{package_name}` is not installed but is required.\n"
            f"Please install it with `{installation_command}`"
        )
