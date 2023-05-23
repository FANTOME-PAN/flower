from abc import abstractmethod, ABC
from typing import Generator

import flwr as fl
import user_messages as usr
from message_handler_registry import register_handler, register_handler_class, ClientMessageHandlerHelper


class IClientMessageHandler(ABC):
    def __init__(self, client):
        ...


class MySAExtension:
    pass

#
# @register_handler_class(MySAExtension, 'SA-setup', 'SA-sharekeys', 'SA-askvec', 'SA-unmask')
# class SecAggProtocolHandler(IClientMessageHandler):
#     def __init__(self, client):
#         super().__init__(client)
#         self.client = client
#
#     @register_handler('SA-setup')
#     def setup(self, task: usr.Task) -> usr.Task:
#         return setup(self, task)
#
#     @register_handler('SA-sharekeys')
#     def share_keys(self, task: usr.Task) -> usr.Task:
#         return share_keys(self, task)
#
#     @register_handler('SA-askvec')
#     def ask_vectors(self, task: usr.Task) -> usr.Task:
#         fit_ins = task.legacy_server_message.fit_ins
#         self.client.fit(fit_ins.parameters, fit_ins.config)
#         return ask_vectors(self, task)
#
#     @register_handler('SA-unmask')
#     def unmask_vectors(self, task: usr.Task) -> usr.Task:
#         return unmask_vectors(self, task)


@register_handler_class(MySAExtension, 'sa_msg')
class SecAggProtocolHandler2(IClientMessageHandler):
    def __init__(self, client):
        super().__init__(client)
        self.client = client

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
        self.client.fit(fit_ins.parameters, fit_ins.config)
        yield ask_vectors(self, task)

        # receive list of dropped clients and active clients and upload relevant info
        task: usr.Task = yield
        yield unmask_vectors(self, task)


class Client(MySAExtension):
    def __init__(self):
        self.shared_states = {}

    @register_handler('fit')
    def fit(self, ins: usr.Task) -> usr.Task:
        ...

    @register_handler('evaluate')
    def evaluate(self, ins: usr.Task) -> usr.Task:
        ...

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


@register_handler_class(Client, 'SA-setup', 'SA-sharekeys', 'SA-askvec', 'SA-unmask')
class SecAggProtocolHandler(IClientMessageHandler):
    def __init__(self, client, quantizier):
        super().__init__(client)
        self.client = client

    @register_handler('SA-setup')
    def setup(self, task: usr.Task) -> usr.Task:
        return setup(self, task)

    @register_handler('SA-sharekeys')
    def share_keys(self, task: usr.Task) -> usr.Task:
        return share_keys(self, task)

    @register_handler('SA-askvec')
    def ask_vectors(self, task: usr.Task) -> usr.Task:
        fit_ins = task.legacy_server_message.fit_ins
        self.client.fit(fit_ins.parameters, fit_ins.config)
        return ask_vectors(self, task)

    @register_handler('SA-unmask')
    def unmask_vectors(self, task: usr.Task) -> usr.Task:
        return unmask_vectors(self, task)




# class FitEvalClientWorkflow(ClientWorkflow):
#
#     def workflow(self: fl.client.NumPyClient) -> Generator[usr.Task, usr.Task, None]:
#         # fit round
#         task: usr.Task = yield
#         fit_ins = task.legacy_server_message.fit_ins
#         yield usr.ClientMessage(fit_res=usr.FitRes(*self.fit(fit_ins.parameters, fit_ins.config)))
#
#         # eval round
#         task: usr.Task = yield
#         eval_ins = task.legacy_server_message.evaluate_ins
#         yield usr.ClientMessage(fit_res=usr.EvaluateRes(*self.evaluate(eval_ins.parameters, eval_ins.config)))
#
#
# class SecAggClientWorkFlow(ClientWorkflow):
#
#     def workflow(self: fl.client.NumPyClient) -> Generator[usr.Task, usr.Task, None]:
#         # setup configurations for SA and upload own public keys
#         task: usr.Task = yield
#         yield setup(self, task)
#
#         # receive other public keys and upload own secret key shares
#         task: usr.Task = yield
#         yield share_keys(self, task)
#
#         # receive other secret key shares, train the model, and upload masked updates
#         task: usr.Task = yield
#         # need training in parallel
#         fit_ins = task.legacy_server_message.fit_ins
#         self.fit(fit_ins.parameters, fit_ins.config)
#         yield ask_vectors(self, task)
#
#         # receive list of dropped clients and active clients and upload relevant info
#         task: usr.Task = yield
#         yield unmask_vectors(self, task)
#

def setup(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...


def share_keys(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...


def ask_vectors(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...


def unmask_vectors(client: fl.client.NumPyClient, task: usr.Task) -> usr.Task:
    ...
