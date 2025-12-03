from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class Normal:
    mean: float
    std: float

@dataclass
class LogNormal:
    mean: float
    std: float

@dataclass
class Uniform:
    low: float
    high: float

@dataclass
class Parameter:
    name: str
    bounds: Optional[Tuple[float, float]] = None
    prior: Optional[Union[Normal, LogNormal, Uniform]] = None
    kernel: Optional[str] = None

@dataclass
class Feed(Parameter):
    pass

@dataclass
class Output:
    name: str
