from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from gordo_components import __version__


__all__ = ["Metadata", "BuildMetadata", "ModelBuildMetadata", "CrossValidationMetaData"]


@dataclass_json
@dataclass
class CrossValidationMetaData:
    scores: Dict[str, Any] = field(default_factory=dict)
    cv_duration_sec: Optional[float] = field(default=None)
    splits: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class ModelBuildMetadata:
    model_offset: int = field(default=0)
    model_creation_date: Optional[str] = field(default=None)
    model_builder_version: str = field(default=__version__)
    data_query_duration_sec: Optional[float] = field(default=None)
    cross_validation: CrossValidationMetaData = field(
        default_factory=CrossValidationMetaData
    )
    model_training_duration_sec: Optional[float] = field(default=None)
    model_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class BuildMetadata:
    model: ModelBuildMetadata = ModelBuildMetadata()


@dataclass_json
@dataclass
class Metadata:
    user_defined: Dict[str, Any] = field(default_factory=dict)
    build_metadata: BuildMetadata = BuildMetadata()
