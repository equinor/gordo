from gordo.machine.dataset.file_system import FileSystem
from gordo.machine.dataset.file_system.adl1 import ADLGen1FileSystem
from gordo.machine.dataset.file_system.adl2 import ADLGen2FileSystem
from gordo.machine.dataset.exceptions import ConfigException
from typing import Optional

DEFAULT_STORAGE_TYPE = "adl1"


def create_storage(storage_type: Optional[str] = None, **kwargs) -> FileSystem:
    """
    Create ``FileSystem`` instance from the config

    Parameters
    ----------
    storage_type: Optional[str]
        Storage type only supported `adl1`, `adl2` values
    kwargs

    Returns
    -------

    """
    if storage_type is None:
        storage_type = DEFAULT_STORAGE_TYPE
    storage: FileSystem
    if storage_type == "adl1":
        if "store_name" not in kwargs:
            kwargs["store_name"] = "dataplatformdlsprod"
        storage = ADLGen1FileSystem.create_from_env(**kwargs)
    elif storage_type == "adl2":
        if "account_name" not in kwargs:
            kwargs["account_name"] = "omniadlseun"
        if "file_system_name" not in kwargs:
            kwargs["file_system_name"] = "dls"
        storage = ADLGen2FileSystem.create_from_env(**kwargs)
    else:
        raise ConfigException("Unknown storage type '%s'" % storage_type)
    return storage
