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
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import ShareKeysPacket, Scalar, Parameters, AskKeysRes
from flwr.common.sa_primitives import secaggplus_primitives
# from flwr_crypto_cpp import create_shares
from flwr.common.sa_primitives.secaggplus_primitives import create_shares
from flwr.common.sa_primitives import quantize, public_key_to_bytes, generate_key_pairs, private_key_to_bytes, \
    bytes_to_public_key, generate_shared_key, encrypt, decrypt
from flwr.common.sa_primitives.weight_arithmetics import *
from flwr.client.abc_sa_client_wrapper import SAClientWrapper
from flwr.common.logger import log
from logging import ERROR, INFO, WARNING
from typing import Dict, List, Tuple


def setup_param(client: SAClientWrapper, setup_param_dict: Dict[str, Scalar]) -> Tuple[bytes, bytes]:
    # Assigning parameter values to object fields
    sec_agg_param_dict = setup_param_dict
    client.sample_num = sec_agg_param_dict['sample_num']
    client.sa_id = sec_agg_param_dict['sa_id']
    client.share_num = sec_agg_param_dict['share_num']
    client.threshold = sec_agg_param_dict['threshold']
    client.clipping_range = sec_agg_param_dict['clipping_range']
    client.target_range = sec_agg_param_dict['target_range']
    client.mod_range = sec_agg_param_dict['mod_range']
    client.max_weights_factor = sec_agg_param_dict['max_weights_factor']

    # Testing , to be removed================================================
    client.test = 0
    if 'test' in sec_agg_param_dict and sec_agg_param_dict['test'] == 1:
        client.test = 1
        client.test_vector_shape = [(sec_agg_param_dict['test_vector_dimension'],)]
        client.test_dropout_value = sec_agg_param_dict['test_dropout_value']
    # End =================================================================

    # key is the sa_id of another client (int)
    # value is the secret share we possess that contributes to the client's secret (bytes)
    client.b_share_dict = {}
    client.sk1_share_dict = {}
    client.shared_key_2_dict = {}
    log(INFO, "SecAgg Stage 0 Completed: Parameters Set Up")
    return ask_keys(client)


def ask_keys(client) -> Tuple[bytes, bytes]:
    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    client.sk1, client.pk1 = generate_key_pairs()
    client.sk2, client.pk2 = generate_key_pairs()
    log(INFO, "SecAgg Stage 1 Completed: Created Key Pairs")
    return public_key_to_bytes(client.pk1), public_key_to_bytes(client.pk2)


def share_keys(client, share_keys_dict: Dict[int, Tuple[str, str]]) -> List[ShareKeysPacket]:
    # Distribute shares for private mask seed and first private key
    # share_keys_dict:
    new_dict: Dict[int, Tuple[bytes, bytes]] = {}
    for k, (pk1, pk2) in share_keys_dict.items():
        new_dict[int(k)] = (pk1.encode('ascii'), pk2.encode('ascii'))
    client.public_keys_dict = new_dict
    # check size is larger than threshold
    if len(client.public_keys_dict) < client.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # check if all public keys received are unique
    pk_list: List[bytes] = []
    for pk1, pk2 in client.public_keys_dict.values():
        pk_list.append(pk1)
        pk_list.append(pk2)
    if len(set(pk_list)) != len(pk_list):
        raise Exception("Some public keys are identical")

    # sanity check that own public keys are correct in dict
    if client.public_keys_dict[client.sa_id][0] != public_key_to_bytes(client.pk1) or \
       client.public_keys_dict[client.sa_id][1] != public_key_to_bytes(client.pk2):
        raise Exception(
            "Own public keys are displayed in dict incorrectly, should not happen!")

    # Generate private mask seed
    client.b = secaggplus_primitives.rand_bytes(32)

    # Create shares
    b_shares = create_shares(
        client.b, client.threshold, client.share_num
    )
    sk1_shares = create_shares(
        private_key_to_bytes(client.sk1), client.threshold, client.share_num
    )

    share_keys_res_list = []

    for idx, p in enumerate(client.public_keys_dict.items()):
        client_sa_id, client_public_keys = p
        if client_sa_id == client.sa_id:
            client.b_share_dict[client.sa_id] = b_shares[idx]
            client.sk1_share_dict[client.sa_id] = sk1_shares[idx]
        else:
            shared_key = generate_shared_key(
                client.sk2, bytes_to_public_key(client_public_keys[1]))
            client.shared_key_2_dict[client_sa_id] = shared_key
            plaintext = secaggplus_primitives.share_keys_plaintext_concat(
                client.sa_id, client_sa_id, b_shares[idx], sk1_shares[idx])
            ciphertext = encrypt(shared_key, plaintext)
            share_keys_packet = ShareKeysPacket(
                source=client.sa_id, destination=client_sa_id, ciphertext=ciphertext)
            share_keys_res_list.append(share_keys_packet)

    log(INFO, "SecAgg Stage 2 Completed: Sent Shares via Packets")
    return share_keys_res_list


def ask_vectors(client, packet_list, fit_ins) -> Parameters:
    # Receive shares and fit model
    available_clients: List[int] = []

    if len(packet_list)+1 < client.threshold:
        raise Exception("Available neighbours number smaller than threshold")

    # decode all packets and verify all packets are valid. Save shares received
    for packet in packet_list:
        source = packet.source
        available_clients.append(source)
        destination = packet.destination
        ciphertext = packet.ciphertext
        if destination != client.sa_id:
            raise Exception(
                "Received packet meant for another user. Not supposed to happen")
        shared_key = client.shared_key_2_dict[source]
        plaintext = decrypt(shared_key, ciphertext)
        try:
            plaintext_source, plaintext_destination, plaintext_b_share, plaintext_sk1_share = \
                secaggplus_primitives.share_keys_plaintext_separate(plaintext)
        except:
            raise Exception(
                "Decryption of ciphertext failed. Not supposed to happen")
        if plaintext_source != source:
            raise Exception(
                "Received packet source is different from intended source. Not supposed to happen")
        if plaintext_destination != destination:
            raise Exception(
                "Received packet destination is different from intended destination. Not supposed to happen")
        client.b_share_dict[source] = plaintext_b_share
        client.sk1_share_dict[source] = plaintext_sk1_share

    # fit client
    # IMPORTANT ASSUMPTION: ASSUME ALL CLIENTS FIT SAME AMOUNT OF DATA
    '''
    fit_res = client.client.fit(fit_ins)
    parameters = fit_res.parameters
    weights = parameters_to_weights(parameters)
    weights_factor = fit_res.num_examples
    '''
    # temporary code=========================================================
    if client.test == 1:
        if client.sa_id % 20 < client.test_dropout_value:
            log(ERROR, "Force dropout due to testing!!")
            raise Exception("Force dropout due to testing")
        weights = weights_zero_generate(client.test_vector_shape)
     # IMPORTANT NEED SOME FUNCTION TO GET CORRECT WEIGHT FACTOR
    # NOW WE HARD CODE IT AS 1
    # Generally, should be fit_res.num_examples

    weights_factor = 1

    # END =================================================================

    # Quantize weight update vector
    quantized_weights = quantize(
        weights, client.clipping_range, client.target_range)

    # weights factor cannoot exceed maximum
    if weights_factor > client.max_weights_factor:
        weights_factor = client.max_weights_factor
        log(WARNING, "weights_factor exceeds allowed range and has been clipped. Either increase max_weights_factor, or train with fewer data. (Or server is performing unweighted aggregation)")

    quantized_weights = weights_multiply(
        quantized_weights, weights_factor)
    quantized_weights = factor_weights_combine(
        weights_factor, quantized_weights)

    dimensions_list: List[Tuple] = [a.shape for a in quantized_weights]

    # add private mask
    private_mask = secaggplus_primitives.pseudo_rand_gen(
        client.b, client.mod_range, dimensions_list)
    quantized_weights = weights_addition(quantized_weights, private_mask)

    for client_id in available_clients:
        # add pairwise mask
        shared_key = generate_shared_key(
            client.sk1, bytes_to_public_key(client.public_keys_dict[client_id][0]))
        # print('shared key length: %d' % len(shared_key))
        pairwise_mask = secaggplus_primitives.pseudo_rand_gen(shared_key, client.mod_range, dimensions_list)
        if client.sa_id > client_id:
            quantized_weights = weights_addition(quantized_weights, pairwise_mask)
        else:
            quantized_weights = weights_subtraction(quantized_weights, pairwise_mask)

    # Take mod of final weight update vector and return to server
    quantized_weights = weights_mod(quantized_weights, client.mod_range)
    log(INFO, "SecAgg Stage 3 Completed: Sent Vectors")
    return ndarrays_to_parameters(quantized_weights)


def unmask_vectors(client, available_clients, dropout_clients) -> Dict[int, bytes]:
    # Send private mask seed share for every avaliable client (including itclient)
    # Send first private key share for building pairwise mask for every dropped client
    if len(available_clients) < client.threshold:
        raise Exception("Available neighbours number smaller than threshold")
    share_dict: Dict[int, bytes] = {}
    for idx in available_clients:
        share_dict[idx] = client.b_share_dict[idx]
    for idx in dropout_clients:
        share_dict[idx] = client.sk1_share_dict[idx]
    log(INFO, "SecAgg Stage 4 Completed: Sent Shares for Unmasking")
    return share_dict
