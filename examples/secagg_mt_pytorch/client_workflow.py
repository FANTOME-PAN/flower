from abc import abstractmethod
from typing import Generator

import flwr as fl
import user_messages as usr
from message_handler_registry import register_handler, ClientMessageHandlerHelper


class MySAExtension:
    @register_handler('setup')
    def setup(self):
        pass


class Client(MySAExtension):
    def __init__(self):
        self.workload_id_to_shared_states = {}

    @register_handler('fit')
    def fit(self, ins: usr.Task) -> usr.Task:
        ...

    @register_handler('evaluate')
    def evaluate(self, ins: usr.Task) -> usr.Task:
        ...

    @register_handler('setup')
    def setup(self, task, shared_states):
        pass

    @register_handler('sa_msg')
    def secagg(self):
        # setup configurations for SA and upload own public keys
        task: usr.Task = yield
        yield setup(self, task)

        # receive other public keys and upload own secret key shares
        task: usr.Task = yield
        yield share_keys(self, task)

        # receive other secret key shares, train the model, and upload masked updates
        task: usr.Task = yield
        # need training in parallel
        fit_ins = task.legacy_server_message.fit_ins
        self.fit(fit_ins.parameters, fit_ins.config)
        yield ask_vectors(self, task)

        # receive list of dropped clients and active clients and upload relevant info
        task: usr.Task = yield
        yield unmask_vectors(self, task)




class FitEvalClientWorkflow(ClientWorkflow):

    def workflow(self: fl.client.NumPyClient) -> Generator[usr.Task, usr.Task, None]:
        # fit round
        task: usr.Task = yield
        fit_ins = task.legacy_server_message.fit_ins
        yield usr.ClientMessage(fit_res=usr.FitRes(*self.fit(fit_ins.parameters, fit_ins.config)))

        # eval round
        task: usr.Task = yield
        eval_ins = task.legacy_server_message.evaluate_ins
        yield usr.ClientMessage(fit_res=usr.EvaluateRes(*self.evaluate(eval_ins.parameters, eval_ins.config)))


class SecAggClientWorkFlow(ClientWorkflow):

    def workflow(self: fl.client.NumPyClient) -> Generator[usr.Task, usr.Task, None]:
        # setup configurations for SA and upload own public keys
        task: usr.Task = yield
        yield setup(self, task)

        # receive other public keys and upload own secret key shares
        task: usr.Task = yield
        yield share_keys(self, task)

        # receive other secret key shares, train the model, and upload masked updates
        task: usr.Task = yield
        # need training in parallel
        fit_ins = task.legacy_server_message.fit_ins
        self.fit(fit_ins.parameters, fit_ins.config)
        yield ask_vectors(self, task)

        # receive list of dropped clients and active clients and upload relevant info
        task: usr.Task = yield
        yield unmask_vectors(self, task)


def setup(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...


def share_keys(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...


def ask_vectors(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...


def unmask_vectors(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...
