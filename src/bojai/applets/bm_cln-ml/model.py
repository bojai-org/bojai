from abc import ABC
import torch
from torch import nn
import numpy as np


# an abstract model used as a base for other models
class Model(ABC):
    def __init__(self):
        super().__init__()


# a logistic regression model. Used for small and medium CLN, could be use for Large
class LogisticRegressionCLN(Model, nn.Module):
    def __init__(self):
        super(LogisticRegressionCLN, self).__init__()
        self.linear = None

    def initialise(self, d):
        self.linear = nn.Linear(d, 1)

    def forward(self, x):  # x is of size n*d
        x = x.to(torch.float32)
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted, 0


## single-layer neural network for number binary classification
class NeuralNetworkCLN(Model, nn.Module):
    def __init__(self):
        super(NeuralNetworkCLN, self).__init__()
        self.linear = None
        self.relu = nn.ReLU()
        self.output_layer = None
        self.sigmoid = nn.Sigmoid()

    def initialise(self, d):
        hidden_size = decide_hs(d)
        self.linear = nn.Linear(d, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = input.to(torch.float32)
        output = self.linear(x)
        output = self.relu(output)
        y_predicted = self.sigmoid(self.output_layer(output))

        return y_predicted, 0


class DeepNeuralNetworkCLN(Model, nn.Module):
    def __init__(self):
        super(DeepNeuralNetworkCLN, self).__init__()
        self.layer1 = None
        self.layer2 = None
        self.relu = nn.ReLU()
        self.output_layer = None
        self.sigmoid = nn.Sigmoid()

    def initialise(self, d):
        hidden_size = decide_hs(d)
        self.model = nn.Sequential(
            nn.Linear(d, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        y_predicted = self.model(input.to(torch.float32))
        return y_predicted, 0


class NeuralNetworkCLNL2(Model, nn.Module):
    def __init__(self):
        super(NeuralNetworkCLNL2, self).__init__()
        self.linear = None
        self.relu = nn.ReLU()
        self.output_layer = None
        self.sigmoid = nn.Sigmoid()

    def initialise(self, d):
        hidden_size = decide_hs(d)
        self.linear = nn.Linear(d, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = input.to(torch.float32)
        output = self.linear(x)
        output = self.relu(output)
        y_predicted = self.sigmoid(self.output_layer(output))
        l2_penalty = sum(torch.norm(param, p=2) ** 2 for param in self.parameters())
        return y_predicted, l2_penalty


class NeuralNetworkCLNL1(Model, nn.Module):
    def __init__(self):
        super(NeuralNetworkCLNL1, self).__init__()
        self.linear = None
        self.relu = nn.ReLU()
        self.output_layer = None
        self.sigmoid = nn.Sigmoid()

    def initialise(self, d):
        hidden_size = decide_hs(d)
        self.linear = nn.Linear(d, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = input.to(torch.float32)
        output = self.linear(x)
        output = self.relu(output)
        y_predicted = self.sigmoid(self.output_layer(output))
        l1_penalty = sum(torch.abs(param).sum() for param in self.parameters())
        return y_predicted, l1_penalty


class NeuralNetworkCLNElasticNet(Model, nn.Module):
    def __init__(self):
        super(NeuralNetworkCLNElasticNet, self).__init__()
        self.linear = None
        self.relu = nn.ReLU()
        self.output_layer = None
        self.sigmoid = nn.Sigmoid()

    def initialise(self, d):
        hidden_size = decide_hs(d)
        self.linear = nn.Linear(d, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = input.to(torch.float32)
        output = self.linear(x)
        output = self.relu(output)
        y_predicted = self.sigmoid(self.output_layer(output))
        l1_penalty = sum(torch.abs(param).sum() for param in self.parameters())
        l2_penalty = sum(torch.norm(param, p=2) ** 2 for param in self.parameters())
        return y_predicted, l1_penalty + l2_penalty


class NeuralNetworkCLNDropout(Model, nn.Module):
    def __init__(self):
        super(NeuralNetworkCLNDropout, self).__init__()
        self.linear = None
        self.relu = nn.ReLU()
        self.output_layer = None
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def initialise(self, d):
        hidden_size = decide_hs(d)
        self.linear = nn.Linear(d, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = input.to(torch.float32)
        output = self.linear(x)
        output = self.relu(output)
        output = self.dropout(output)
        y_predicted = self.sigmoid(self.output_layer(output))
        return y_predicted, 0


class kNN(Model, nn.Module):
    def __init__(self):
        super(kNN, self).__init__()
        self.linear = None

    def initialise(self, input, output):
        self.data = torch.tensor(input, dtype=torch.float32)
        self.labels = torch.tensor(output, dtype=torch.float32)

    def forward(self, input):
        n_samples, d_features = input.shape
        k = 1  # Change if tuning

        # Compute pairwise distances
        dist = torch.cdist(input, self.data)

        # Get k nearest neighbors
        knn_indices = dist.topk(k, largest=False).indices
        knn_labels = self.labels[knn_indices]

        # Count votes
        votes = torch.zeros((n_samples, 2), dtype=torch.int32)
        votes[:, 0] = (knn_labels == 0).sum(dim=1)
        votes[:, 1] = (knn_labels == 1).sum(dim=1)

        # Pick the label with more votes
        final_predictions = torch.argmax(votes, dim=1)

        # Make shape and dtype compatible with BCELoss
        return final_predictions.unsqueeze(1).float(), 0


def decide_hs(d):
    if d < 4:
        return 4
    if d < 16:
        return d  # Keep hidden size equal to input size for small dimensions
    if d < 64:
        return int(1.5 * d)  # More gradual scaling
    return min(int(1.5 * d), 256)  # Upper cap for very high dimensions
