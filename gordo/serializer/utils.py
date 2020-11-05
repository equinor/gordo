from .constants import ALLOWED_IMPORT_PATH


def validate_import_path(import_path: str) -> bool:
    """
    Security validation for import paths allowed in serializer

    Examples
    --------
    >>> validate_import_path("gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector")
    True
    >>> validate_import_path("os.rmdir")
    False
    >>> validate_import_path("..module.MaliciousCode")
    False
    """
    if not import_path.find(".") == 0:
        for allowed_import in ALLOWED_IMPORT_PATH:
            if import_path.find(allowed_import) == 0:
                return True
    return False
