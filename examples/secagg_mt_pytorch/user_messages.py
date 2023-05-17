from dataclasses import dataclass
from typing import List, Dict, Union, Optional

import numpy as np

from flwr.common import Scalar


@dataclass
class FitIns:
    parameters: List[np.ndarray]
    config: Dict[str, Scalar]


@dataclass
class FitRes:
    parameters: List[np.ndarray]
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class EvaluateIns:
    parameters: List[np.ndarray]
    config: Dict[str, Scalar]


@dataclass
class EvaluateRes:
    loss: float
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class ServerMessage:
    fit_ins: Optional[FitIns] = None
    evaluate_ins: Optional[EvaluateIns] = None


@dataclass
class ClientMessage:
    fit_res: Optional[FitRes] = None
    evaluate_res: Optional[EvaluateRes] = None


@dataclass
class SecureAggregationMessage:
    named_arrays: Dict[str, Union[np.ndarray, List[np.ndarray]]] = None
    named_bytes: Dict[str, Union[bytes, List[bytes]]] = None
    named_scalars: Dict[str, Union[Scalar, List[Scalar]]] = None


@dataclass
class Task:
    legacy_server_message: Optional[ServerMessage] = None
    legacy_client_message: Optional[ClientMessage] = None
    secure_aggregation_message: Optional[SecureAggregationMessage] = None
    new_message_flag = True
