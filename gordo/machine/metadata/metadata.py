from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from gordo import __version__


__all__ = [
    "Metadata",
    "BuildMetadata",
    "ModelBuildMetadata",
    "CrossValidationMetaData",
    "DatasetBuildMetadata",
]


@dataclass_json
@dataclass
class CrossValidationMetaData:
    scores: Dict[str, Any] = field(default_factory=dict)
    cv_duration_sec: Optional[float] = None
    splits: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class ModelBuildMetadata:
    model_offset: int = 0
    model_creation_date: Optional[str] = None
    model_builder_version: str = __version__
    cross_validation: CrossValidationMetaData = field(
        default_factory=CrossValidationMetaData
    )
    model_training_duration_sec: Optional[float] = None
    model_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class DatasetBuildMetadata:
    query_duration_sec: Optional[float] = None  # How long it took to get the data
    dataset_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class BuildMetadata:
    model: ModelBuildMetadata = field(default_factory=ModelBuildMetadata)
    dataset: DatasetBuildMetadata = field(default_factory=DatasetBuildMetadata)


@dataclass_json
@dataclass
class Metadata:
    user_defined: Dict[str, Any] = field(default_factory=dict)
    build_metadata: BuildMetadata = field(default_factory=BuildMetadata)
