# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: flwr/proto/task.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15\x66lwr/proto/task.proto\x12\nflwr.proto\"\x12\n\x04Task\x12\n\n\x02id\x18\x01 \x01(\t\"\x19\n\x06Result\x12\x0f\n\x07task_id\x18\x01 \x01(\tb\x06proto3')



_TASK = DESCRIPTOR.message_types_by_name['Task']
_RESULT = DESCRIPTOR.message_types_by_name['Result']
Task = _reflection.GeneratedProtocolMessageType('Task', (_message.Message,), {
  'DESCRIPTOR' : _TASK,
  '__module__' : 'flwr.proto.task_pb2'
  # @@protoc_insertion_point(class_scope:flwr.proto.Task)
  })
_sym_db.RegisterMessage(Task)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), {
  'DESCRIPTOR' : _RESULT,
  '__module__' : 'flwr.proto.task_pb2'
  # @@protoc_insertion_point(class_scope:flwr.proto.Result)
  })
_sym_db.RegisterMessage(Result)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TASK._serialized_start=37
  _TASK._serialized_end=55
  _RESULT._serialized_start=57
  _RESULT._serialized_end=82
# @@protoc_insertion_point(module_scope)
