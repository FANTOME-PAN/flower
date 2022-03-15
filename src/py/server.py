import argparse
from typing import Dict

from flwr.common.logger import log
from flwr.server.grpc_server.grpc_server import start_insecure_grpc_server

import torch
import torchvision

import flwr as fl
import data_partition
from collections import OrderedDict
from utils import test
from flwr.server.strategy import FedAvg
from model import build_model
# from flwr.server.strategy.sec_agg_fedavg import SecAggFedAvg
from strategy import ReducedSecAgg

DEFAULT_SERVER_ADDRESS = "localhost:8099"

DEVICE = torch.device("cuda")


def start_server(exp_name=None,
                 server_address=DEFAULT_SERVER_ADDRESS,
                 rounds=1,
                 epochs=10,
                 batch_size=32,
                 sample_fraction=1.0,
                 min_sample_size=2,
                 min_num_clients=2,
                 log_host=None,
                 log_dict='logs/',
                 alpha=1e-9):
    if not exp_name:
        exp_name = f"epoch_{epochs}_" \
                   f"clients_{min_num_clients}"

    # Configure logger
    fl.common.logger.configure("server", host=log_host)

    # Load evaluation data
    _, testset = data_partition.load_data()

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()

    strategy = ReducedSecAgg(
        fraction_fit=sample_fraction,
        min_fit_clients=min_sample_size,
        min_eval_clients=min_sample_size,
        min_available_clients=min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=generate_config(epochs, batch_size),
        on_evaluate_config_fn=generate_config(epochs, batch_size),
        sec_agg_param_dict={
                            "min_num": min_sample_size,
                            "share_num": 5,
                            "threshold": 4,
                            'max_weights_factor': 1,
                            'target_range': 1 << 24,
                            'clipping_range': 16,
                            'alpha': alpha
                            })

    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    print(f"Starting gRPC server on {server_address}...")
    grpc_server = start_insecure_grpc_server(
        client_manager=server.client_manager(),
        server_address=server_address,
        max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH,
    )

    # Fit model
    print("Fitting the model...")
    hist = server.fit(num_rounds=rounds, sec_agg=1)

    # Write training history to file
    f = open(log_dict + exp_name, "w")
    f.write(hist.__repr__())
    f.close()

    # Stop the gRPC server
    grpc_server.stop(None)


def generate_config(epochs, batch_size):
    def fit_config(round: int) -> Dict[str, str]:
        print(f"Configuring round {round}...")
        return {
            "epoch_global": str((round - 1) * epochs),
            "epochs": str(epochs),
            "batch_size": str(batch_size),
        }

    return fit_config


def get_eval_fn(testset: torchvision.datasets.CIFAR10):
    """Returns an evaluation function for centralized (server-side) evaluation."""

    def evaluate(weights: fl.common.Weights):
        """Use the entire CIFAR-10 test set for evaluation."""
        model = build_model()
        if len(weights) < len(model.state_dict().items()):
            print(1111111111)
            keys = [k for k, v in model.state_dict().items() if v.shape != ()]
        else:
            keys = model.state_dict().keys()
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(keys, weights)}
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = test(net=model, testloader=testloader, device=DEVICE, log_progress=False)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument("--server_address", type=str, default=DEFAULT_SERVER_ADDRESS,
                        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds of federated learning (default: 1)")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of available clients used for fit/evaluate (default: 1.0)")
    parser.add_argument("--min_sample_size", type=int, default=2,
                        help="Minimum number of clients used for fit/evaluate (default: 2)")
    parser.add_argument("--min_num_clients", type=int, default=2,
                        help="Minimum number of available clients needed for sampling (default: 2)")
    parser.add_argument("--log_host", type=str, help="Log server address (no default)")
    parser.add_argument("--log_dir", type=str, help="Log directory. (default:logs/)", default='logs/')
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs each client will train for (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of samples per batch each client will use (default: 32)")
    parser.add_argument("--exp_name", type=str,
                        help="Name of the experiment you are running (no default)")
    parser.add_argument("--alpha", type=float, default=1e-9,
                        help="alpha value")
    args, _ = parser.parse_known_args()

    start_server(exp_name=args.exp_name,
                 server_address=args.server_address,
                 rounds=args.rounds,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 sample_fraction=args.sample_fraction,
                 min_sample_size=args.min_sample_size,
                 min_num_clients=args.min_num_clients,
                 log_host=args.log_host,
                 log_dict=args.log_dir,
                 alpha=args.alpha)


if __name__ == "__main__":
    main()
