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
"""ProtoBuf serialization and deserialization."""


from typing import Any, List, Tuple, cast, Union

from flwr.proto.transport_pb2 import (
    ClientMessage,
    Code,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    Status,
)

from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.server import Server
from flwr_experimental.baseline import config

from . import typing

#  === ServerMessage message ===


def server_message_to_proto(server_message: typing.ServerMessage) -> ServerMessage:
    """Serialize `ServerMessage` to ProtoBuf."""
    if server_message.get_properties_ins is not None:
        return ServerMessage(
            get_properties_ins=get_properties_ins_to_proto(
                server_message.get_properties_ins,
            )
        )
    if server_message.get_parameters_ins is not None:
        return ServerMessage(
            get_parameters_ins=get_parameters_ins_to_proto(
                server_message.get_parameters_ins,
            )
        )
    if server_message.fit_ins is not None:
        return ServerMessage(
            fit_ins=fit_ins_to_proto(
                server_message.fit_ins,
            )
        )
    if server_message.evaluate_ins is not None:
        return ServerMessage(
            evaluate_ins=evaluate_ins_to_proto(
                server_message.evaluate_ins,
            )
        )
    raise Exception("No instruction set in ServerMessage, cannot serialize to ProtoBuf")


def server_message_from_proto(
    server_message_proto: ServerMessage,
) -> typing.ServerMessage:
    """Deserialize `ServerMessage` from ProtoBuf."""
    field = server_message_proto.WhichOneof("msg")
    if field == "get_properties_ins":
        return typing.ServerMessage(
            get_properties_ins=get_properties_ins_from_proto(
                server_message_proto.get_properties_ins,
            )
        )
    if field == "get_parameters_ins":
        return typing.ServerMessage(
            get_parameters_ins=get_parameters_ins_from_proto(
                server_message_proto.get_parameters_ins,
            )
        )
    if field == "fit_ins":
        return typing.ServerMessage(
            fit_ins=fit_ins_from_proto(
                server_message_proto.fit_ins,
            )
        )
    if field == "evaluate_ins":
        return typing.ServerMessage(
            evaluate_ins=evaluate_ins_from_proto(
                server_message_proto.evaluate_ins,
            )
        )
    raise Exception(
        "Unsupported instruction in ServerMessage, cannot deserialize from ProtoBuf"
    )


#  === ClientMessage message ===


def client_message_to_proto(client_message: typing.ClientMessage) -> ClientMessage:
    """Serialize `ClientMessage` to ProtoBuf."""
    if client_message.get_properties_res is not None:
        return ClientMessage(
            get_properties_res=get_properties_res_to_proto(
                client_message.get_properties_res,
            )
        )
    if client_message.get_parameters_res is not None:
        return ClientMessage(
            get_parameters_res=get_parameters_res_to_proto(
                client_message.get_parameters_res,
            )
        )
    if client_message.fit_res is not None:
        return ClientMessage(
            fit_res=fit_res_to_proto(
                client_message.fit_res,
            )
        )
    if client_message.evaluate_res is not None:
        return ClientMessage(
            evaluate_res=evaluate_res_to_proto(
                client_message.evaluate_res,
            )
        )
    raise Exception("No instruction set in ClientMessage, cannot serialize to ProtoBuf")


def client_message_from_proto(
    client_message_proto: ClientMessage,
) -> typing.ClientMessage:
    """Deserialize `ClientMessage` from ProtoBuf."""
    field = client_message_proto.WhichOneof("msg")
    if field == "get_properties_res":
        return typing.ClientMessage(
            get_properties_res=get_properties_res_from_proto(
                client_message_proto.get_properties_res,
            )
        )
    if field == "get_parameters_res":
        return typing.ClientMessage(
            get_parameters_res=get_parameters_res_from_proto(
                client_message_proto.get_parameters_res,
            )
        )
    if field == "fit_res":
        return typing.ClientMessage(
            fit_res=fit_res_from_proto(
                client_message_proto.fit_res,
            )
        )
    if field == "evaluate_res":
        return typing.ClientMessage(
            evaluate_res=evaluate_res_from_proto(
                client_message_proto.evaluate_res,
            )
        )
    raise Exception(
        "Unsupported instruction in ClientMessage, cannot deserialize from ProtoBuf"
    )


#  === Parameters message ===


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """Serialize `Parameters` to ProtoBuf."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """Deserialize `Parameters` from ProtoBuf."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === ReconnectIns message ===


def reconnect_ins_to_proto(ins: typing.ReconnectIns) -> ServerMessage.ReconnectIns:
    """Serialize `ReconnectIns` to ProtoBuf."""
    if ins.seconds is not None:
        return ServerMessage.ReconnectIns(seconds=ins.seconds)
    return ServerMessage.ReconnectIns()


def reconnect_ins_from_proto(msg: ServerMessage.ReconnectIns) -> typing.ReconnectIns:
    """Deserialize `ReconnectIns` from ProtoBuf."""
    return typing.ReconnectIns(seconds=msg.seconds)


# === DisconnectRes message ===


def disconnect_res_to_proto(res: typing.DisconnectRes) -> ClientMessage.DisconnectRes:
    """Serialize `DisconnectRes` to ProtoBuf."""
    reason_proto = Reason.UNKNOWN
    if res.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif res.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif res.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.DisconnectRes(reason=reason_proto)


def disconnect_res_from_proto(msg: ClientMessage.DisconnectRes) -> typing.DisconnectRes:
    """Deserialize `DisconnectRes` from ProtoBuf."""
    if msg.reason == Reason.RECONNECT:
        return typing.DisconnectRes(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.DisconnectRes(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.DisconnectRes(reason="WIFI_UNAVAILABLE")
    return typing.DisconnectRes(reason="UNKNOWN")


# === GetParameters messages ===


def get_parameters_ins_to_proto(
    ins: typing.GetParametersIns,
) -> ServerMessage.GetParametersIns:
    """Serialize `GetParametersIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetParametersIns(config=config)


def get_parameters_ins_from_proto(
    msg: ServerMessage.GetParametersIns,
) -> typing.GetParametersIns:
    """Deserialize `GetParametersIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetParametersIns(config=config)


def get_parameters_res_to_proto(
    res: typing.GetParametersRes,
) -> ClientMessage.GetParametersRes:
    """Serialize `GetParametersRes` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        return ClientMessage.GetParametersRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.GetParametersRes(
        status=status_msg, parameters=parameters_proto
    )


def get_parameters_res_from_proto(
    msg: ClientMessage.GetParametersRes,
) -> typing.GetParametersRes:
    """Deserialize `GetParametersRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    return typing.GetParametersRes(status=status, parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize `FitIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize `FitIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize `FitIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        return ClientMessage.FitRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.FitRes(
        status=status_msg,
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize `FitRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        status=status,
        parameters=parameters,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === GetProperties messages ===


def get_properties_ins_to_proto(
    ins: typing.GetPropertiesIns,
) -> ServerMessage.GetPropertiesIns:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetPropertiesIns(config=config)


def get_properties_ins_from_proto(
    msg: ServerMessage.GetPropertiesIns,
) -> typing.GetPropertiesIns:
    """Deserialize `GetPropertiesIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetPropertiesIns(config=config)


def get_properties_res_to_proto(
    res: typing.GetPropertiesRes,
) -> ClientMessage.GetPropertiesRes:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        return ClientMessage.GetPropertiesRes(status=status_msg)
    properties_msg = properties_to_proto(res.properties)
    return ClientMessage.GetPropertiesRes(status=status_msg, properties=properties_msg)


def get_properties_res_from_proto(
    msg: ClientMessage.GetPropertiesRes,
) -> typing.GetPropertiesRes:
    """Deserialize `GetPropertiesRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    properties = properties_from_proto(msg.properties)
    return typing.GetPropertiesRes(status=status, properties=properties)


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize `EvaluateIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize `EvaluateIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize `EvaluateIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        return ClientMessage.EvaluateRes(status=status_msg)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.EvaluateRes(
        status=status_msg,
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize `EvaluateRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        status=status,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === Status messages ===


def status_to_proto(status: typing.Status) -> Status:
    """Serialize `Status` to ProtoBuf."""
    code = Code.OK
    if status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        code = Code.FIT_NOT_IMPLEMENTED
    if status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        code = Code.EVALUATE_NOT_IMPLEMENTED
    return Status(code=code, message=status.message)


def status_from_proto(msg: Status) -> typing.Status:
    """Deserialize `Status` from ProtoBuf."""
    code = typing.Code.OK
    if msg.code == Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if msg.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if msg.code == Code.FIT_NOT_IMPLEMENTED:
        code = typing.Code.FIT_NOT_IMPLEMENTED
    if msg.code == Code.EVALUATE_NOT_IMPLEMENTED:
        code = typing.Code.EVALUATE_NOT_IMPLEMENTED
    return typing.Status(code=code, message=msg.message)


# === Properties messages ===


def properties_to_proto(properties: typing.Properties) -> Any:
    """Serialize `Properties` to ProtoBuf."""
    proto = {}
    for key in properties:
        proto[key] = scalar_to_proto(properties[key])
    return proto


def properties_from_proto(proto: Any) -> typing.Properties:
    """Deserialize `Properties` from ProtoBuf."""
    properties = {}
    for k in proto:
        properties[k] = scalar_from_proto(proto[k])
    return properties


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize `Metrics` to ProtoBuf."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize `Metrics` from ProtoBuf."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


# === Scalar messages ===


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize `Scalar` to ProtoBuf."""

    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize `Scalar` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.Scalar, scalar)


# === Secure Aggregation Messages ===


# Server side
def sa_server_msg_carrier_to_proto(ins: typing.SAServerMessageCarrier)\
        -> ServerMessage.SAMessageCarrier:
    np_arr_lst = None if ins.numpy_ndarray_list is None else parameters_to_proto(
        ndarrays_to_parameters(ins.numpy_ndarray_list))
    str2scalar = None if ins.str2scalar is None else metrics_to_proto(ins.str2scalar)
    params = None if ins.parameters is None else parameters_to_proto(ins.parameters)
    fit_ins = None if ins.fit_ins is None else fit_ins_to_proto(ins.fit_ins)
    return ServerMessage.SAMessageCarrier(
        identifier=ins.identifier,
        ndarray_list=np_arr_lst,
        str2scalar=str2scalar,
        bytes_list=ins.bytes_list,
        parameters=params,
        fit_ins=fit_ins
    )


def sa_server_msg_carrier_from_proto(proto: ServerMessage.SAMessageCarrier)\
        -> typing.SAServerMessageCarrier:
    np_arr_lst = parameters_to_ndarrays(proto.ndarray_list)
    str2scalar = metrics_from_proto(proto.str2scalar)
    params = parameters_from_proto(proto.parameters)
    fit_ins = fit_ins_from_proto(proto.fit_ins)
    return typing.SAServerMessageCarrier(
        identifier=proto.identifier,
        numpy_ndarray_list=np_arr_lst,
        str2scalar=str2scalar,
        bytes_list=proto.bytes_list,
        parameters=params,
        fit_ins=fit_ins
    )


def check_sa_error(msg: ClientMessage.SAMessageCarrier):
    if msg.error_msg != '':
        raise Exception(msg.error_msg)


# Client side
def sa_client_msg_carrier_to_proto(ins: typing.SAClientMessageCarrier)\
        -> ClientMessage.SAMessageCarrier:
    np_arr_lst = None if ins.numpy_ndarray_list is None else parameters_to_proto(
        ndarrays_to_parameters(ins.numpy_ndarray_list))
    str2scalar = None if ins.str2scalar is None else metrics_to_proto(ins.str2scalar)
    params = None if ins.parameters is None else parameters_to_proto(ins.parameters)
    fit_res = None if ins.fit_res is None else fit_res_to_proto(ins.fit_res)
    return ClientMessage.SAMessageCarrier(
        identifier=ins.identifier,
        ndarray_list=np_arr_lst,
        str2scalar=str2scalar,
        bytes_list=ins.bytes_list,
        parameters=params,
        fit_res=fit_res
    )


def sa_client_msg_carrier_from_proto(proto: ClientMessage.SAMessageCarrier)\
        -> typing.SAClientMessageCarrier:
    np_arr_lst = parameters_to_ndarrays(proto.ndarray_list)
    str2scalar = metrics_from_proto(proto.str2scalar)
    params = parameters_from_proto(proto.parameters)
    fit_res = fit_res_from_proto(proto.fit_res)
    return typing.SAClientMessageCarrier(
        identifier=proto.identifier,
        numpy_ndarray_list=np_arr_lst,
        str2scalar=str2scalar,
        bytes_list=proto.bytes_list,
        parameters=params,
        fit_res=fit_res
    )
