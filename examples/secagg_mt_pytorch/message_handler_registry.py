import inspect

from typing import Dict

import user_messages as usr


__reg = type('', (), {'functions': {}, 'type2handlers': None})()


def register_handler(message_type: str):
    def register(func):
        __reg.functions.setdefault(message_type, []).append(func)
        __reg.type2handlers = None
        return func

    return register


def get_handlers(cls_type: type) -> Dict[str, str]:
    if __reg.type2handlers is None:
        type2handlers: Dict[type, set] = {}
        for msg_type, func_lst in __reg.functions.items():
            for func in func_lst:
                cls = func.__globals__[func.__qualname__.split('.')[0]]
                type2handlers.setdefault(cls, set()).add((msg_type, func.__name__))
        __reg.type2handlers = type2handlers
    type2handlers = __reg.type2handlers
    handlers = set()
    for _type, _set in type2handlers.items():
        if issubclass(cls_type, _type):
            handlers.update(_set)
    ret = {}
    for msg_type, func_name in handlers:
        if msg_type in ret and ret[msg_type] != func_name:
            raise ValueError(f"A handler for the \"{msg_type}\" message type is already registered.")
        ret[msg_type] = func_name
    return ret


class ClientMessageHandlerHelper:
    class _GeneratorAdapter:
        def __init__(self, generator_func):
            self.generator_func = generator_func
            self.wf = generator_func()

        def reset(self):
            self.wf.close()
            self.wf = self.generator_func()

        def __call__(self, task: usr.Task) -> usr.Task:
            if task.new_message_flag:
                self.reset()
            try:
                next(self.wf)
            except StopIteration:
                self.reset()
                next(self.wf)
            return self.wf.send(task)

    def __init__(self, client):
        self.client = client
        msg2fname = get_handlers(type(client))
        self.handlers = {}
        for msg_type, func_name in msg2fname.items():
            func = client.__getattribute__(func_name)
            if inspect.isgeneratorfunction(func):
                self.handlers[msg_type] = ClientMessageHandlerHelper._GeneratorAdapter(func)
            else:
                self.handlers[msg_type] = func

    def handle(self, message_type, message: usr.Task) -> usr.Task:
        if message_type in self.handlers:
            return self.handlers[message_type](message)
        raise ValueError(f'{self.client.__class__}: cannot handle the message type {message_type}.')


