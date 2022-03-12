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
from torchvision.models import mobilenet
from utils import train, test, get_sample_result
import config as c

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

        self.sample_data = load_sample_data()

    def get_parameters(self):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Load model parameters
        self.set_parameters(parameters)

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        epoch_global = int(config["epoch_global"])

        start_epoch = epoch_global + 1
        end_epoch = start_epoch + epochs - 1

        # Train the model
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        train(self.model, trainloader, device=self.device, start_epoch=start_epoch, end_epoch=end_epoch)

        # Run evaluation
        testloader = DataLoader(self.testset, batch_size=32, shuffle=False)
        loss, accuracy = test(self.model, testloader, device=self.device)
        # print('client' + str(self.cid), accuracy)
        return self.get_parameters(), len(self.trainset), {'cid': self.cid}

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
    model = c.model(10)
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
