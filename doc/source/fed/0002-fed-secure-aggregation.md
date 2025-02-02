---
fed-number: 0002
title: secure aggregation
authors: ["@FANTOME-PAN"]
creation-date: 2023-04-25
last-updated: 2023-04-27
status: provisional
---

# FED Template

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#summary)
- [Proposal](#proposal)
  - [Data types for SA](#data-types-for-sa)
  - [Server-side components](#server-side-components)
  - [Client-side components](#client-side-components)

## Summary

The current Flower framework does not have built-in modules for Secure Aggregation (SA).
However, flower users may want to use SA in their FL solutions or
implement their own SA protocols easily.

Based on the previous SA implementation, I intend to build the SA 
for flower on the Driver API.



## Proposal

### Data types for SA

Judging from the SecAgg protocol, the SecAgg+ protocol, the LightSecAgg protocol,
and the FastSecAgg protocol, the following fields can better facilitate
SA implementations.

1. bytes, List of bytes

    SA protocols often use encryption and send ciphertext in bytes.
Besides, cryptography-related information, such as public keys, are normally stored as bytes.
Sharing these info will require transmitting bytes.

    Currently, both FitIns and FitRes contain one dictionary field,
mapping strings to scalars (including bytes).
    Though it is possible to store lists of bytes in the dictionary using tricks,
it can be easier to implement SA if TaskIns and TaskRes have fields supporting bytes and lists of bytes

2. arrays

    In many protocols, the server and the clients need to send 
additional but necessary information to complete SA.
These info are usually single or multiple lists of integers or floats.
We now need to store them in the parameters field.


Considering all above, if possible, I would suggest adding a more general dictionary field,
i.e., Dict[str, Union[Dict[str, LScalar], LScalar]],
where LScalar = Union[Scalar, List[Scalar]]

Alternatively, we can have multiple dictionary fields in addition to the config/metrics dictionary, including:

1. Dict[str, Union[np.ndarray, List[np.ndarray]]]
2. Dict[str, Union[bytes, List[bytes]]]
3. Dict[str, Scalar]

Example code:

``` protobuf
message Task {
  Node producer = 1;
  Node consumer = 2;
  string created_at = 3;
  string delivered_at = 4;
  string ttl = 5;
  repeated string ancestry = 6;
  SecAggMsg sec_agg = 7;
  
  ServerMessage legacy_server_message = 101 [ deprecated = true ];
  ClientMessage legacy_client_message = 102 [ deprecated = true ];
}

message SecAggMsg {
  message Arrays {
    message Plural {
      repeated bytes value = 1;
    }
    oneof value {
      bytes singular = 1;
      Plural plural = 2;
    }
  }
  message Bytes {
    message Plural {
      repeated bytes value = 1;
    }
    oneof value {
      bytes singular = 1;
      Plural plural = 2;
    }
  }
  message Scalars {
    message Plural {
      repeated Scalar value = 1;
    }
    oneof value {
      Scalar singular = 1;
      Plural plural = 2;
    }
  }

  map<string, Arrays> named_arrays = 1;
  map<string, Bytes> named_bytes = 2;
  map<string, Scalars> named_scalars = 3;
}
```


### Server-side components

The server actively coordinates the SA protocols.
Its responsibilities include:
1. help broadcast SA configs for initialisation.
2. forward messages from one client to another.
3. gathering information from clients to obtain aggregate output.

In short, other then serving as a relay, the server is a controller and decryptor.
It controls the workflow. Since SA protocols are rather different from each other,
we may want to allow customising the workflow, i.e., allowing users to define 
arbitrary rounds of communication in a single FL fit round.

Example code as follows.

User-facing classes:

```python
@dataclass
class ServerMessage:
    fit_ins: FitIns = None


@dataclass
class SecureAggregationMessage:
    named_arrays: Dict[str, Union[np.ndarray, List[np.ndarray]]] = None
    named_bytes: Dict[str, Union[bytes, List[bytes]]] = None
    named_scalars: Dict[str, Union[Scalar, List[Scalar]]] = None


@dataclass
class NodeTask:
    legacy_server_message: ServerMessage = None
    secure_aggregation_message: SecureAggregationMessage = None
```

Workflows (generators):

```python
def workflow_without_sec_agg(strategy: Strategy):
    # configure fit
    sampled_node_ids: List[int] = yield
    fit_ins: FitIns = FitIns(parameters=strategy.parameters, config={})
    task = NodeTask(legacy_server_message=ServerMessage(fit_ins))
    yield {node_id: task for node_id in sampled_node_ids}

    # aggregate fit
    node_messages: Dict[int, NodeTask] = yield
    print(f'updating parameters with received messages {node_messages}...')
    # todo


def workflow_with_sec_agg(strategy: Strategy):
    sampled_node_ids: List[int] = yield

    yield request_keys_ins(sampled_node_ids)
    node_messages: Dict[int, NodeTask] = yield

    yield share_keys_ins(node_messages)
    node_messages: Dict[int, NodeTask] = yield

    yield request_parameters_ins(node_messages)
    node_messages: Dict[int, NodeTask] = yield

    yield request_key_shares_ins(sampled_node_ids, node_messages)
    node_messages: Dict[int, NodeTask] = yield
    print(f'trying to decrypt and update parameters...')
    # todo
```

Driver:

```python
import random
import time
from typing import List, Dict

from flwr.common import ndarrays_to_parameters
from flwr.driver import Driver
from flwr.proto import driver_pb2, task_pb2, node_pb2
from task import Net, get_parameters
from workflows import workflow_with_sec_agg, NodeTask


def user_task_to_proto(task: NodeTask) -> task_pb2.Task:
    ...


def user_task_from_proto(proto: task_pb2.Task) -> NodeTask:
    ...


# -------------------------------------------------------------------------- Driver SDK
driver = Driver(driver_service_address="0.0.0.0:9091", certificates=None)
# -------------------------------------------------------------------------- Driver SDK

anonymous_client_nodes = True
num_client_nodes_per_round = 1
sleep_time = 1
num_rounds = 1
parameters = ndarrays_to_parameters(get_parameters(net=Net()))
workflow = workflow_with_sec_agg(None)  # should specify a strategy instance

# -------------------------------------------------------------------------- Driver SDK
driver.connect()
# -------------------------------------------------------------------------- Driver SDK

for server_round in range(num_rounds):
    print(f"Commencing server round {server_round + 1}")

    # List of sampled node IDs in this round
    sampled_node_ids: List[int] = []

    # Sample node ids
    if anonymous_client_nodes:
        # If we're working with anonymous clients, we don't know their identities, and
        # we don't know how many of them we have. We, therefore, have to assume that
        # enough anonymous client nodes are available or become available over time.
        #
        # To schedule a TaskIns for an anonymous client node, we set the node_id to 0
        # (and `anonymous` to True)
        # Here, we create an array with only zeros in it:
        sampled_node_ids = [0] * num_client_nodes_per_round
    else:
        # If our client nodes have identiy (i.e., they are not anonymous), we can get
        # those IDs from the Driver API using `get_nodes`. If enough clients are
        # available via the Driver API, we can select a subset by taking a random
        # sample.
        #
        # The Driver API might not immediately return enough client node IDs, so we
        # loop and wait until enough client nodes are available.
        while True:
            # Get a list of node ID's from the server
            get_nodes_req = driver_pb2.GetNodesRequest()

            # ---------------------------------------------------------------------- Driver SDK
            get_nodes_res: driver_pb2.GetNodesResponse = driver.get_nodes(
                req=get_nodes_req
            )
            # ---------------------------------------------------------------------- Driver SDK

            all_node_ids: List[int] = get_nodes_res.node_ids
            print(f"Got {len(all_node_ids)} node IDs")

            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_node_ids = random.sample(
                    all_node_ids, num_client_nodes_per_round
                )
                break

            time.sleep(3)

    # Log sampled node IDs
    print(f"Sampled {len(sampled_node_ids)} node IDs: {sampled_node_ids}")
    time.sleep(sleep_time)

    node_responses = sampled_node_ids

    for _ in workflow:
        ins: Dict[int, NodeTask] = workflow.send(node_responses)
        task_ins_list: List[task_pb2.TaskIns] = []
        # Schedule a task for all sampled nodes
        for node_id, user_task in ins.items():
            new_task = user_task_to_proto(user_task)
            new_task_ins = task_pb2.TaskIns(
                task_id="",  # Do not set, will be created and set by the DriverAPI
                group_id="",
                workload_id="",
                task=task_pb2.Task(
                    producer=node_pb2.Node(
                        node_id=0,
                        anonymous=True,
                    ),
                    consumer=node_pb2.Node(
                        node_id=node_id,
                        anonymous=anonymous_client_nodes,  # Must be True if we're working with anonymous clients
                    ),
                    legacy_server_message=new_task.legacy_server_message,
                    sec_agg=new_task.sec_agg
                ),
            )
            task_ins_list.append(new_task_ins)

        push_task_ins_req = driver_pb2.PushTaskInsRequest(task_ins_list=task_ins_list)

        # ---------------------------------------------------------------------- Driver SDK
        push_task_ins_res: driver_pb2.PushTaskInsResponse = driver.push_task_ins(
            req=push_task_ins_req
        )
        # ---------------------------------------------------------------------- Driver SDK

        print(
            f"Scheduled {len(push_task_ins_res.task_ids)} tasks: {push_task_ins_res.task_ids}"
        )

        time.sleep(sleep_time)

        # Wait for results, ignore empty task_ids
        task_ids: List[str] = [
            task_id for task_id in push_task_ins_res.task_ids if task_id != ""
        ]
        all_task_res: List[task_pb2.TaskRes] = []
        while True:
            pull_task_res_req = driver_pb2.PullTaskResRequest(
                node=node_pb2.Node(node_id=0, anonymous=True),
                task_ids=task_ids,
            )

            # ------------------------------------------------------------------ Driver SDK
            pull_task_res_res: driver_pb2.PullTaskResResponse = driver.pull_task_res(
                req=pull_task_res_req
            )
            # ------------------------------------------------------------------ Driver SDK

            task_res_list: List[task_pb2.TaskRes] = pull_task_res_res.task_res_list
            print(f"Got {len(task_res_list)} results")

            time.sleep(sleep_time)

            all_task_res += task_res_list
            
            # in secure aggregation, this may changed to a timer:
            # when reaching time limit, the server will assume the nodes have lost connection.
            if len(all_task_res) == len(task_ids):
                break

        # "Aggregate" results
        node_responses = {task_res.task.producer: user_task_from_proto(task_res.task) for task_res in all_task_res}
        print(f"Received {len(node_responses)} results")
    
        time.sleep(sleep_time)

    # Repeat

# -------------------------------------------------------------------------- Driver SDK
driver.disconnect()
# -------------------------------------------------------------------------- Driver SDK

```

### Client-side components

The key responsibilities of a client are:
1. generate (cryptography-related) information
2. sharing information via the server
3. encrypt its output
4. help the server decrypt the aggregate output

In summary, a client is an encryptor. It requires additional information from 
other encryptors for initialisation and also provides other encryptors with its information.
Then, it can independently encrypt its output. 
In the end of the fit round, it provides the server with necessary information that allows
and only allows the server to decrypt aggregate output, learning nothing of individual outputs.

