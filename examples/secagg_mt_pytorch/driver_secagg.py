import random
import time
from typing import List, Dict

from flwr.common import ndarrays_to_parameters
from flwr.driver import Driver
from flwr.proto import driver_pb2, task_pb2, node_pb2
from task import Net, get_parameters
from workflows import workflow_with_sec_agg
import user_messages as usr


def user_task_to_proto(task: usr.Task) -> task_pb2.Task:
    ...


def user_task_from_proto(proto: task_pb2.Task) -> usr.Task:
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

    while True:
        try:
            next(workflow)
            ins: Dict[int, usr.Task] = workflow.send(node_responses)
        except StopIteration:
            break
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
