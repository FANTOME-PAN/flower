from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np

from flwr.common import Scalar


@dataclass
class FitInstruction:
    parameters: List[np.ndarray]
    config: Dict[str, Scalar]


@dataclass
class ServerMessage:
    fit_ins: FitInstruction = None


@dataclass
class SecureAggregationMessage:
    named_arrays: Dict[str, Union[np.ndarray, List[np.ndarray]]] = None
    named_bytes: Dict[str, Union[bytes, List[bytes]]] = None
    named_scalars: Dict[str, Union[Scalar, List[Scalar]]] = None


@dataclass
class Task:
    legacy_server_message: ServerMessage = None
    secure_aggregation_message: SecureAggregationMessage = None
