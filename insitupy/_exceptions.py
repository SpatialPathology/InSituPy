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
        
class XeniumDataRepeatedCropError(Exception):
    """Exception raised if it is attempted to crop a
    XeniumData object multiple times with the same cropping window.

    Args:
        xlim: 
            Limits on x-axis.
        ylim: 
            Limits on y-axis.
    """

    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        self.message = f"\nXeniumData object has been cropped with the same limits before:\n" \
            f"xlim -> {xlim}\n" \
            f"ylim -> {ylim}" 
        super().__init__(self.message)
        
class XeniumDataMissingObject(Exception):
    """Exception raised if a certain object is not available in the XeniumData object.
    Maybe it has to be read first

    Args:
        name: 
            Name of object that is searched for.
    """

    def __init__(self, name):
        self.name = name
        self.message = f"\nXeniumData object does not contain object `{name}`.\n" \
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
        
class FileNotFoundError(Exception):
    """Exception raised if a certain file is not found.

    Args:
        name: 
            General name of the file (e.g. metadata)
        filename: 
            File name of the file that was not found (e.g. experiment.xenium)
        directory: 
            Directory in which the file was searched.
    """

    def __init__(self, 
                 name: str, 
                 filename: str,
                 directory: str
                 ):
        self.message = f"{name} file was not found. Searched for file name {filename} in following directory:\n{directory}"
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
    """Exception raised if a certain modality is not found by XeniumData read modules.

    Args:
        modality: 
            Name of the modality (e.g. matrix)
    """

    def __init__(self, 
                 modality: str, 
                 ):
        self.message = f"No `{modality}` modality found."
        super().__init__(self.message)
        
class InvalidFileTypeError(Exception):
    def __init__(self, allowed_types, received_type, message=None):
        allowed_types = convert_to_list(allowed_types)
        if message is None:
            message = f"Invalid file type. Allowed file types: {', '.join(allowed_types)}. Received: {received_type}"
        self.message = message
        super().__init__(self.message)
        