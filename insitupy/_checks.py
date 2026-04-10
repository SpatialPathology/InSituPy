from ._exceptions import MissingPackageError


def try_import(package_name, import_as=None, installation_command=None):
    """Import *package_name*, raising :exc:`MissingPackageError` if it is not installed.

    Args:
        package_name: Top-level name of the package to import.
        import_as: Unused alias parameter (reserved for future use).
        installation_command: Optional install hint shown in the error message.

    Returns:
        The imported module object.

    Raises:
        MissingPackageError: If the package cannot be imported.
    """
    try:
        return __import__(package_name)
    except ImportError:
        raise MissingPackageError(
            package_name,
            installation_command)