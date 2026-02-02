import torch.nn as nn
import math
import torch

class SurrogateModel(nn.Module):
    def __init__(self,  hidden_dim=256, n_layers=2, input_dim = 6, output_dim = 2):

        super().__init__()

        layers_list = []
        for i in range(n_layers):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *[layer for layer in layers_list],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
