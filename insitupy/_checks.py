from ._exceptions import MissingPackageError


def try_import(package_name, import_as=None, installation_command=None):
    try:
        return __import__(package_name)
    except ImportError:
        raise MissingPackageError(
            package_name,
            installation_command)