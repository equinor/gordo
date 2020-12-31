from typing import Optional, Any, List, Dict
from pydantic import BaseModel


class ResourceRequirements(BaseModel):
    requests: Optional[Dict[str, Any]]
    limits: Optional[Dict[str, Any]]


class ConfigMapKeySelector(BaseModel):
    key: str
    value: str


class SecretKeySelector(BaseModel):
    key: str
    name: str


class EnvVarSource(BaseModel):
    configMapKeyRef: Optional[ConfigMapKeySelector]
    secretKeyRef: Optional[SecretKeySelector]


class EnvVar(BaseModel):
    name: str
    value: Optional[str]
    valueFrom: Optional[EnvVarSource]


class ObjectMeta(BaseModel):
    labels: Optional[Dict[str, Any]]


class CSIVolumeSource(BaseModel):
    driver: str
    readOnly: bool = False
    fsType: Optional[str]
    volumeAttributes: Optional[Dict[str, Any]]


class Volume(BaseModel):
    name: str
    csi: Optional[CSIVolumeSource]


class VolumeMount(BaseModel):
    name: str
    readOnly: Optional[bool]
    mountPath: str


class PodRuntime(BaseModel):
    image: str
    resources: Optional[ResourceRequirements]
    metadata: Optional[ObjectMeta]
    env: Optional[List[EnvVar]]
    volumeMounts: Optional[List[VolumeMount]]


class RemoteLogging(BaseModel):
    enable: bool = False


class BuilderPodRuntime(PodRuntime):
    remote_logging: RemoteLogging
