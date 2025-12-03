from dataclasses import dataclass, field
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
class GaussianSmoothing:
    sigma: float = 1.0

@dataclass
class Feed(Parameter):
    smoothing: GaussianSmoothing = field(default_factory=GaussianSmoothing)

@dataclass
class Output:
    name: str
