# def __getattr__(name):
#     # Check if spatialdata package is available
#     try:
#         import spatialdata
#     except ImportError:
#         raise ImportError(
#             "The spatialdata module requires the 'spatialdata' package. "
#             "Install it with: pip install spatialdata"
#         )

#     # Try each submodule in order
#     submodules = ['structured', 'spatialdata']

#     for submodule_name in submodules:
#         try:
#             submodule = __import__(f'insitupy.spatialdata.{submodule_name}',
#                                    fromlist=[name])
#             if hasattr(submodule, name):
#                 return getattr(submodule, name)
#         except (ImportError, AttributeError):
#             continue

#     raise AttributeError(f"module 'insitupy.spatialdata' has no attribute '{name}'")

from .convert import convert_to_spatialdata
from .structured import StructuredSpatialData
