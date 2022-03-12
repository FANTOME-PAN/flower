import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
from flwr_experimental.baseline.dataset.dataset import create_partitioned_dataset
# from data_partition_utils import create_partitioned_dataset
import os
from torch.utils.data import Subset


def load_data():
    """Loads CIFAR-10 (training and test set)."""
    data_root = "./data/cifar-10"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return trainset, testset


def load_sample_data():
    data_root = "./data/cifar-10"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    torch.manual_seed(0)
    indices = torch.randperm(len(testset))[:10]
    sample_set = Subset(testset, indices)
    return sample_set


class PartitionedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], int(self.Y[idx]))


def load_local_partitioned_data(client_id, iid_fraction: float, num_partitions: int):
    """Creates a dataset for each worker, which is a partition of a larger dataset."""
    if os.path.exists(f'partitions/cifar10_cid{client_id}.pth'):
        torch_partition_trainset, torch_partition_testset = torch.load(
            f'partitions/cifar10_part{client_id}/{num_partitions}.pth')
        return torch_partition_trainset, torch_partition_testset
    # Each worker loads the entire dataset, and then selects its partition
    # determined by its `client_id` (happens internally below)
    trainset, testset = load_data()

    train_loader = DataLoader(trainset, batch_size=len(trainset))
    test_loader = DataLoader(testset, batch_size=len(testset))

    (x_train, y_train), (x_test, y_test) = next(iter(train_loader)), next(iter(test_loader))
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()

    (train_partitions, test_partitions), _ = create_partitioned_dataset(
        ((x_train, y_train), (x_test, y_test)), iid_fraction, num_partitions)

    x_train, y_train = train_partitions[client_id]
    torch_partition_trainset = PartitionedDataset(torch.Tensor(x_train), y_train)
    x_test, y_test = test_partitions[client_id]
    torch_partition_testset = PartitionedDataset(torch.Tensor(x_test), y_test)
    torch.save((torch_partition_trainset, torch_partition_testset),
               f'partitions/cifar10_part{client_id}-{num_partitions}.pth')
    return torch_partition_trainset, torch_partition_testset

