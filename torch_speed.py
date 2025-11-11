import torch as torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_features, input_shape, output_shape, hidden_layers, hidden_layer_shape, validation, learning_rate):
        super().__init__()
        self.in_features = in_features
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.hidden_layers_shape = hidden_layer_shape
        self.validation = validation
        self.learning_rate = learning_rate
        
    def _scale_data(self):
        if self.in_features < 0 | self.in_features > 1:
            in_max = np.max(self.in_features)
            self.scaled_features = self.in_features / in_max
        else:
            self.scaled_features = self.in_features
        
        if self.validation < 0 | self.validation > 1:
            val_max = np.max(self.validation)
            self.scaled_validation = self.validation / val_max
        else:
            self.scaled_validation = self.validation
        return self.scaled_validation, self.scaled_features
            
            
    def _tensorize(self):
        in_tensor_is = torch.is_tensor(self.scaled_features)
        if in_tensor_is == True:
            self.data_in = self.scaled_features 
        elif in_tensor_is == False:
            self.data_in = torch.tensor(self.scaled_features)
        
        val_tensor_is = torch.is_tensor(self.scaled_validation)
        if in_tensor_is == True:
            self.data_val = self.scaled_validation 
        elif in_tensor_is == False:
            self.data_val = torch.tensor(self.scaled_validation)
        return self.data_val, self.data_in
    
    def _layers(self):
        layers = []
        layers.append(nn.Linear(self.input_shape, self.hidden_layers_shape)) #input layer
        layers.append(nn.ReLU)
        for _ in range(self.hidden_layers - 1): #hidden layers
            layers.append(nn.Linear(self.hidden_layers_shape, self.hidden_layers_shape))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_layers_shape, self.output_shape)) #output layer
        self.model = nn.Sequential(*layers)
    
    def _model_params(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
    
    def _send_to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device: {device}")
        
        self.model.to(device) 
        self.data_in.to(device)
        self.data_val.to(device)     
    
    def _initialize(self):
        _scale_data   
    
        
    
    
        


    
