from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Generator

import user_messages as usr
from flwr.common import Parameters


@dataclass
class Strategy:
    parameters: Parameters


class FlowerWorkflowGenerator:
    def __init__(self, strategy: Strategy, num_rounds=1e9):
        self.num_rounds = num_rounds
        self.strategy = strategy

    @abstractmethod
    def configure_tasks(self, node_messages: Union[List[int], Dict[int, usr.Task]]) \
            -> Dict[int, usr.Task]:
        ...

    @abstractmethod
    def aggregate_tasks(self, node_messages: Dict[int, usr.Task]):
        ...

    def generate_workflow(self) \
            -> Generator[Dict[int, usr.Task], Union[List[int], Dict[int, usr.Task]], None]:
        node_messages = yield
        for _ in range(self.num_rounds):
            ins_dict = self.configure_tasks(node_messages)
            if ins_dict is None:
                break
            yield ins_dict
            node_messages = yield
            self.aggregate_tasks(node_messages)


def workflow_without_sec_agg(strategy: Strategy) \
        -> Generator[Dict[int, usr.Task], Union[List[int], Dict[int, usr.Task]], None]:
    # configure fit
    sampled_node_ids: List[int] = yield
    fit_ins = usr.FitIns(parameters=strategy.parameters, config={})
    task = usr.Task(legacy_server_message=usr.ServerMessage(fit_ins=fit_ins))
    yield {node_id: task for node_id in sampled_node_ids}

    # aggregate fit
    node_messages: Dict[int, usr.Task] = yield
    print(f'updating parameters with received messages {node_messages}...')
    # todo


def workflow_with_sec_agg(strategy: Strategy) \
        -> Generator[Dict[int, usr.Task], Union[List[int], Dict[int, usr.Task]], None]:
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




