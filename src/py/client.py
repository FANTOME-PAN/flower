import argparse
import grpc
import timeit

import numpy as np
import torch
import torchvision
import flwr as fl

from collections import OrderedDict
from typing import Optional
from torch.utils.data import DataLoader
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

from data_partition import load_local_partitioned_data, load_sample_data
from model import build_model
from utils import train, test, get_sample_result
from flwr.common.sec_agg.sec_agg_primitives import weights_subtraction
from flwr.common.parameter import weights_to_parameters, parameters_to_weights

DEFAULT_SERVER_ADDRESS = "localhost:8099"


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        model,
        testset,
        trainset,
        device):
        self.cid = cid
        self.model = model
        self.testset = testset
        self.trainset = trainset
        self.device = device
        self.cnt = 0
        self.sample_data = load_sample_data()
        self.available_keys = [k for k, v in self.model.state_dict().items() if v.shape != ()]

    def get_parameters(self):
        ret = [val.detach().cpu().numpy() for _, val in self.model.state_dict().items() if val.shape != ()]
        for i, arr in enumerate(ret):
            if arr.max() > 3:
                print(f"Client {self.cid}: exceeding range in f{self.available_keys[i]} with max {arr.max()}")
        return ret

    def set_parameters(self, parameters):
        # print(f"parameters: {type(parameters)} of {type(parameters[0])}")
        params_dict = zip(self.available_keys, [o for o in parameters if o.shape != ()])
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict if v.shape != ()})
        # print([o.shape for o in parameters][:8])
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        # Load model parameters
        parameters = [o for o in parameters if o.shape != ()]
        self.set_parameters(parameters)

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        epoch_global = int(config["epoch_global"])

        start_epoch = epoch_global + 1
        end_epoch = start_epoch + epochs - 1
        print(f"client {self.cid} starts to fit")
        # Train the model
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        train(self.model, trainloader, device=self.device, start_epoch=start_epoch, end_epoch=end_epoch, max_iter=3)
        diff = weights_subtraction(self.get_parameters(), parameters)
        # Run evaluation
        # testloader = DataLoader(self.testset, batch_size=32, shuffle=False)
        # loss, accuracy = test(self.model, testloader, device=self.device)
        # print('client' + str(self.cid), accuracy)
        return diff, 1, {'cid': self.cid}

    def evaluate(self, parameters, config):
        # Load model parameters
        self.set_parameters(parameters)

        # Run evaluation
        testloader = DataLoader(self.testset, batch_size=32, shuffle=False)
        loss, accuracy = test(self.model, testloader, device=self.device)
        print('client' + str(self.cid), accuracy)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def start_client(client_id, num_partitions, iid_fraction=1.0,
                 server_address=DEFAULT_SERVER_ADDRESS, log_host=None, exp_name=None):
    # Configure logger
    fl.common.logger.configure(f"client_{client_id}", host=log_host)

    print(f"Loading data for client {client_id}")
    trainset, testset = load_local_partitioned_data(
        client_id=client_id,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions)

    # Load model and data
    cuda_id = 0
    for cuda_id, lst in enumerate(np.array_split(np.arange(num_partitions), torch.cuda.device_count())):
        if client_id in lst:
            break
    device_name = 'cuda:' + str(cuda_id)
    DEVICE = torch.device(device_name)
    model = build_model()
    model.to(DEVICE)

    # Start client
    print(f"Starting client {client_id}")
    client = CifarClient(client_id, model, testset, trainset, DEVICE)

    print(f"Connecting to {server_address}")

    try:
        # There's no graceful shutdown when gRPC server terminates, so we try/except
        fl.client.start_numpy_client(server_address, client)
    except grpc._channel._MultiThreadedRendezvous:
        print(f"Client {client_id}: shutdown")


def main():
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS,
                        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})")
    parser.add_argument("--cid", type=int, required=True, help="Client CID (no default)")
    parser.add_argument("--num_partitions", type=int, required=True,
                        help="Total number of clients participating in training")
    parser.add_argument("--iid_fraction", type=float, nargs="?", const=1.0,
                        help="Fraction of data [0,1] that is independent and identically distributed.")
    parser.add_argument("--log_host", type=str, help="Log server address")
    parser.add_argument("--exp_name", type=str, help="Friendly experiment name")

    args, _ = parser.parse_known_args()
    start_client(client_id=args.cid,
                 num_partitions=args.num_partitions,
                 iid_fraction=args.iid_fraction,
                 log_host=args.log_host,
                 server_address=args.server_address,
                 exp_name=args.exp_name)


if __name__ == "__main__":
    main()
