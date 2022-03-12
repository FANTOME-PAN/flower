import numpy as np
from coordinators import *
from model import ResNet

''' define how clients are partitioned into groups
  [[0, 1], [2, 3]] means clients with cid 0 and 1
  are allocated to the same group, clients with cid 2 and 3
  are in the other group. '''
groups = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

''' define iid fraction between groups in different hierarchical
  layers. [0.0, 1.0] means the iid_fractions between groups in
  the first layer (i.e., between [0, 1] group and [2, 3] group)
  are 0.0, and the iid_fractions between groups in the second layer 
  (i.e., between client with cid 0 and 1, between client with cid 2
  and 3) are 1.0 '''
iid_fractions = [0.0, 0.4, 0.8]

''' partitions of the network layers. [0, 50, 100] means the first 
  50 layers are shared across all clients, the next 50 layers are
  shared between group in the second hierarchical layers, the rest
  are layers individual to each client.
'''
net_partitions = [0, 90, 108, 120]

''' portions of parameters which will be used for parameters aggregation
  the n_th row represent the portions of parameter for the n_th client
  for example, [0.4, 0.4, 0.1, 0.1] on the first row means the first
  client will take 0.4 * parameters from the second client and 0.1 *
  parameters from the 3rd and 4th client, this is only for FedWeight
  and FedWHier
'''
portions = np.array([[[0.3, 0.3, 0.2, 0.2],
                     [0.3, 0.3, 0.2, 0.2],
                     [0.2, 0.2, 0.3, 0.3],
                     [0.2, 0.2, 0.3, 0.3]],
                    [[0.4, 0.4, 0.1, 0.1],
                     [0.4, 0.4, 0.1, 0.1],
                     [0.1, 0.1, 0.4, 0.4],
                     [0.1, 0.1, 0.4, 0.4]]])

''' aggregation strategy used by the server. Can be one of those:
  FedHier: Hierachical weights aggregation, hyper-parameters defined
  above will be used
  FedWeight: Weighted average aggregation, portions array will be used
  FedWHier: weighted hierachical aggregation, combining FedHier and FedWeight
  FedAvg: Default aggregation strategy defined in the flower package
  FedIndiv: Equivalent to training model locally on each client
'''
strategy = 'FedWHier'

model = ResNet.ResNet18

coordinator = GridWeightCoordinator(groups)




