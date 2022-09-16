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
"""(De-)serialization tests."""

from typing import Union, cast

from flwr.common import typing
from flwr.common.typing import SAClientMessageCarrier, SAServerMessageCarrier
from flwr.proto import transport_pb2 as pb2

from .serde import (
    sa_client_msg_carrier_from_proto,
    sa_client_msg_carrier_to_proto,
    sa_server_msg_carrier_from_proto,
    sa_server_msg_carrier_to_proto,
    scalar_from_proto,
    scalar_to_proto,
    status_from_proto,
    status_to_proto,
)


def test_serialisation_deserialisation() -> None:
    """Test if the np.ndarray is identical after (de-)serialization."""

    # Prepare
    scalars = [True, b"bytestr", 3.14, 9000, "Hello"]

    for scalar in scalars:
        # Execute
        scalar = cast(Union[bool, bytes, float, int, str], scalar)
        serialized = scalar_to_proto(scalar)
        actual = scalar_from_proto(serialized)

        # Assert
        assert actual == scalar


def test_secure_aggregation():
    """Test if data stored in SA carriers keep unchanged before and after (de-)serialization."""
    import numpy as np
    inputs = [
        SAServerMessageCarrier('2'),
        SAServerMessageCarrier('2231', numpy_ndarray_list=[np.arange(100)]),
        SAServerMessageCarrier('sfas_', str2scalar={'dsa': b'safsqqq'}, bytes_list=[b'aaqq  a', b'test1123!~']),
        SAServerMessageCarrier('aa', parameters=Parameters([b'parameters', b'params'], 'meters')),
        SAServerMessageCarrier('21dsva', fit_ins=FitIns(Parameters([b'a'], 'type'), dict())),
    ]

    def check_lst(lst1, lst2):
        assert len(lst1) == len(lst2)
        for o1, o2 in zip(lst1, lst2):
            assert o1 == o2

    for o in inputs:
        serialized = sa_server_msg_carrier_to_proto(o)
        actual = sa_server_msg_carrier_from_proto(serialized)
        assert actual.identifier == o.identifier

    inputs = [
        SAClientMessageCarrier('2'),
        SAClientMessageCarrier('2231', numpy_ndarray_list=[np.arange(100)]),
        SAClientMessageCarrier('sfas_', str2scalar={'dsa': b'safsqqq'}, bytes_list=[b'aaqq  a', b'test1123!~']),
        SAClientMessageCarrier('aa', parameters=Parameters([b'parameters', b'params'], 'meters')),
        SAClientMessageCarrier('21dsva', fit_res=FitRes(Parameters([b'a'], 'type'), None)),
    ]

    for o in inputs:
        serialized = sa_client_msg_carrier_to_proto(o)
        actual = sa_client_msg_carrier_from_proto(serialized)
        assert actual.identifier == o.identifier


def test_status_to_proto() -> None:
    """Test status message (de-)serialization."""

    # Prepare
    code_msg = pb2.Code.OK
    status_msg = pb2.Status(code=code_msg, message="Success")

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status_msg = status_to_proto(status=status)

    # Assert
    assert actual_status_msg == status_msg


def test_status_from_proto() -> None:
    """Test status message (de-)serialization."""

    # Prepare
    code_msg = pb2.Code.OK
    status_msg = pb2.Status(code=code_msg, message="Success")

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status = status_from_proto(msg=status_msg)

    # Assert
    assert actual_status == status
