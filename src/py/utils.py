from collections import OrderedDict, defaultdict
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import SparsePCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
from tqdm import tqdm
from model import MobileNet
from torchvision import datasets


def train(
    net: MobileNet,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    start_epoch: int,
    end_epoch: int,
    log_progress: bool = True,
    max_iter=10000):
    """Trains a network on provided data from `start_epoch` to `end_epoch` incl. (the training loop).
    @param net:
    @param trainloader:
    @param device:
    @param start_epoch:
    @param end_epoch:
    @param log_progress:
    @param max_iter:
    @return: list of (train loss, train acc) for each epoch
    """

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)

    print(f"Training from epoch(s) {start_epoch} to {end_epoch} w/ {len(trainloader)} batches each.", flush=True)
    results = []

    # Train the network
    for epoch in range(start_epoch, end_epoch + 1):  # loop over the dataset multiple times, last epoch inclusive
        total_loss, total_correct, n_samples = 0.0, 0.0, 0
        pbar = tqdm(trainloader, desc=f'TRAIN Epoch {epoch}') if log_progress else trainloader
        cnt = 0
        for data in pbar:
            if cnt >= max_iter:
                break
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Collected training loss and accuracy statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            if log_progress:
                pbar.set_postfix({
                    "train_loss": total_loss / n_samples,
                    "train_acc": total_correct / n_samples
                })
            cnt += 1

        results.append((total_loss / n_samples, total_correct / n_samples))

    return results


def test(
    net: MobileNet,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    log_progress: bool = True):
    """Evaluates the network on test data.
    @param net:
    @param testloader:
    @param device:
    @param log_progress:
    @return: avg loss, acc
    """
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            # Collected testing loss and accuracy statistics
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    return total_loss / n_samples, total_correct / n_samples


def get_sample_result(net, partitions, sample_data, device):
    activations = []

    def hook(model, input, output):
        activations.append(output.cpu().detach().numpy())

    module_names = []
    layer_names = list(net.state_dict().keys())
    for idx in partitions[1:]:
        module_names.append('.'.join(layer_names[idx - 1].split('.')[:2]))

    handles = []
    for name, layer in net.named_modules():
        if name in module_names:
            handles.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        for data in sample_data:
            output = net(data[0].to(device))
            activations.append(output.cpu().detach().numpy())

    for handle in handles:
        handle.remove()

    return activations

