# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]
MetricsAggregationFn = Callable[[List[Tuple[int, Metrics]]], Metrics]

Config = Dict[str, Scalar]
Properties = Dict[str, Scalar]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str


@dataclass
class GetParametersIns:
    """Parameters request for a client."""

    config: Config


@dataclass
class GetParametersRes:
    """Response when asked to return parameters."""

    status: Status
    parameters: Parameters


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    status: Status
    loss: float
    num_examples: int
    metrics: Dict[str, Scalar]


@dataclass
class GetPropertiesIns:
    """Properties request for a client."""

    config: Config


@dataclass
class GetPropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties


@dataclass
class ReconnectIns:
    """ReconnectIns message from server to client."""

    seconds: Optional[int]


@dataclass
class DisconnectRes:
    """DisconnectRes message from client to server."""

    reason: str


@dataclass
class ServerMessage:
    """ServerMessage is a container used to hold one instruction message."""

    get_properties_ins: Optional[GetPropertiesIns] = None
    get_parameters_ins: Optional[GetParametersIns] = None
    fit_ins: Optional[FitIns] = None
    evaluate_ins: Optional[EvaluateIns] = None


@dataclass
class ClientMessage:
    """ClientMessage is a container used to hold one result message."""

    get_properties_res: Optional[GetPropertiesRes] = None
    get_parameters_res: Optional[GetParametersRes] = None
    fit_res: Optional[FitRes] = None
    evaluate_res: Optional[EvaluateRes] = None



import json


class SAMessage:
    def __init__(self, json_string: str = None):
        if json_string is not None:
            self.from_json(json_string)

    def to_json(self):
        # stat = self.__dict__.copy()
        # _rebuild = []
        # for k in stat:
        #     if isinstance(stat[k], SAMessage):
        #         _rebuild.append(k)
        #         stat[k] = stat[k].to_json()
        # stat['_rebuild'] = _rebuild
        # return json.dumps(stat)
        return json.dumps(self.__dict__)

    def from_json(self, json_string: str):
        self.__dict__.update(json.loads(json_string))
        # if '_rebuild' in self.__dict__:
        #     for k in self._rebuild:
        #         self.__dict__[k] = SAMessage(self.__dict__[k])
        return self


@dataclass
class SAServerMessageCarrier:
    identifier: str
    numpy_ndarray_list: Optional[List[np.ndarray]] = None
    str2scalar: Optional[Dict[str, Scalar]] = None
    bytes_list: Optional[List[bytes]] = None
    parameters: Optional[Parameters] = None
    fit_ins: Optional[FitIns] = None
    sa_msg: SAMessage = SAMessage()


@dataclass
class SAClientMessageCarrier:
    identifier: str
    numpy_ndarray_list: Optional[np.ndarray] = None
    str2scalar: Optional[Dict[str, Scalar]] = None
    bytes_list: Optional[List[bytes]] = None
    parameters: Optional[Parameters] = None
    fit_res: Optional[FitRes] = None
    sa_msg: SAMessage = SAMessage()


@dataclass
class ShareKeysPacket:
    source: int
    destination: int
    ciphertext: bytes


@dataclass
class AskKeysRes:
    pk1: bytes
    pk2: bytes


@dataclass
class SetupParamIns:
    sec_agg_param_dict: Dict[str, Scalar]


@dataclass
class SetupParamRes:
    pass


@dataclass
class AskKeysIns:
    pass


@dataclass
class AskKeysRes:
    """Ask Keys Stage Response from client to server"""

    pk1: bytes
    pk2: bytes


@dataclass
class ShareKeysIns:
    public_keys_dict: Dict[int, AskKeysRes]


@dataclass
class ShareKeysPacket:
    source: int
    destination: int
    ciphertext: bytes


@dataclass
class ShareKeysRes:
    share_keys_res_list: List[ShareKeysPacket]


@dataclass
class AskVectorsIns:
    ask_vectors_in_list: List[ShareKeysPacket]
    fit_ins: FitIns


@dataclass
class AskVectorsRes:
    parameters: Parameters


@dataclass
class UnmaskVectorsIns:
    available_clients: List[int]
    dropout_clients: List[int]


@dataclass
class UnmaskVectorsRes:
    share_dict: Dict[int, bytes]


