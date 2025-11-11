import torch as torch
import torch.nn as nn
import pandas


class MLP(nn.Module):
    def __init__(self, in_features, input_shape, output_shape, hidden_layers, hidden_layer_shape):
        super().__init__()
        self.in_features = in_features
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.hidden_layers_shape = hidden_layer_shape

    def _scale_data(self):

    def _tensorize(self):
        tensor_is = torch.is_tensor(self.in_features)
        if tensor_is == True:
            self.data = self.in_features 
        elif tensor_is == False:
            self.data = torch.tensor(self.in_features)
        return self.data
    
    def _layers(self):
        self.input_layer = nn.Linear(self.input_shape, self.hidden_layers_shape)
        for i in range(self.hidden_layers - 1):


    
