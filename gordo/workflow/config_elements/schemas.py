from typing import Optional, Union
from pydantic import BaseModel

ValueType = Union[str, int, float, bool]


class RefValue(BaseModel):
    name: str
    key: str


class ValueFrom(BaseModel):
    configMapKeyRef: Optional[RefValue]
    secretKeyRef: Optional[RefValue]


class CustomEnv(BaseModel):
    name: str
    value: Optional[ValueType]
    valueFrom: Optional[ValueFrom]
