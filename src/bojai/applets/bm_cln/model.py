from abc import ABC
import torch
from torch import nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTConfig,
    BertConfig,
    VisionEncoderDecoderConfig,
)


# an abstract model used as a base for other models
class Model(ABC):
    def __init__(self):
        super().__init__()


# a logistic regression model. Used for small and medium CLN
class LogisticRegressionCLN(Model, nn.Module):
    def __init__(self):
        super(LogisticRegressionCLN, self).__init__()
        self.linear = None

    def initialise(self, d):
        self.linear = nn.Linear(d, 1)

    def forward(self, x):  # x is of size n*d
        x = x.to(torch.float32)
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
