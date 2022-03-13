# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Secure Aggregation (SecAgg) Bonawitz et al.

Paper: https://eprint.iacr.org/2017/281.pdf
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple
from functools import reduce
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.common.sec_agg.sec_agg_primitives import weights_addition
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.strategy.sec_agg_strategy import SecAggStrategy

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""


class ReducedSecAgg(FedAvg, SecAggStrategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        sec_agg_param_dict: Dict[str, Scalar] = {}
    ) -> None:

        FedAvg.__init__(self, fraction_fit=fraction_fit,
                        fraction_eval=fraction_eval,
                        min_fit_clients=min_fit_clients,
                        min_eval_clients=min_eval_clients,
                        min_available_clients=min_available_clients,
                        eval_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        on_evaluate_config_fn=on_evaluate_config_fn,
                        accept_failures=accept_failures,
                        initial_parameters=initial_parameters)
        self.sec_agg_param_dict = sec_agg_param_dict
        self.cached_params = None

    def get_sec_agg_param(self) -> Dict[str, int]:
        return self.sec_agg_param_dict.copy()

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        assert self.cached_params is None
        self.cached_params = parameters
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        assert self.cached_params is not None
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        ret = aggregate(weights_results)
        ret = weights_addition(ret, parameters_to_weights(self.cached_params))
        self.cached_params = None
        return weights_to_parameters(ret), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    # weighted_weights = [
    #     [layer * num_examples for layer in weights] for weights, num_examples in results
    # ]
    weights = [o for o, _ in results]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weights)
    ]
    return weights_prime


def weighted_loss_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    return sum(weighted_losses) / num_total_evaluation_examples

