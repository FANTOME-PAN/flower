from typing import List, Union, Dict, Tuple
import random
import time
from dataclasses import dataclass
from flwr.driver import Driver
from flwr.common import ndarrays_to_parameters, serde, Parameters, Scalar
from flwr.proto import driver_pb2, task_pb2, node_pb2, transport_pb2

from task import Net, get_parameters, set_parameters
import numpy as np
import user_facing_messages as usr


@dataclass
class Strategy:
    parameters: Parameters


def workflow_without_sec_agg(strategy: Strategy):
    # configure fit
    sampled_node_ids: List[int] = yield
    fit_ins = usr.FitInstruction(parameters=strategy.parameters, config={})
    task = usr.Task(legacy_server_message=usr.ServerMessage(fit_ins=fit_ins))
    yield {node_id: task for node_id in sampled_node_ids}

    # aggregate fit
    node_messages: Dict[int, usr.Task] = yield
    print(f'updating parameters with received messages {node_messages}...')
    # todo


def workflow_with_sec_agg(strategy: Strategy):
    sampled_node_ids: List[int] = yield

    yield request_keys_ins(sampled_node_ids)
    node_messages: Dict[int, usr.Task] = yield

    yield share_keys_ins(node_messages)
    node_messages: Dict[int, usr.Task] = yield

    yield request_parameters_ins(node_messages)
    node_messages: Dict[int, usr.Task] = yield

    yield request_key_shares_ins(sampled_node_ids, node_messages)
    node_messages: Dict[int, usr.Task] = yield
    print(f'trying to decrypt and update parameters...')
    # todo


def request_keys_ins(ids):
    pass


def share_keys_ins(res):
    pass


def request_parameters_ins(res):
    pass


def request_key_shares_ins(ids, res):
    pass




