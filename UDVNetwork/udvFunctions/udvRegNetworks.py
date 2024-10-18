# Name: Regression neural network structure
# Function: Define neural netwroks for regression
#===========================================================
# Necessary package
import torch
import torch.nn as nn
from .udvDiagonalLayer import D_singleConnection
#===========================================================
#==========================START============================
# Define the UDV network (Single UDV)
# For UDV, UDV-s, UDV-v1, UDV-v2
class UDV_net_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()    
        self.fc1 = nn.Linear(self.num_input, self.num_hidden_1, bias = False) 
        self.diag1 = D_singleConnection(num_in_Diag = self.num_hidden_1)
        self.fc2 = nn.Linear(self.num_hidden_1, self.num_output, bias = False)
           
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.diag1(x)
        x = self.fc2(x).squeeze(1)
        return x
    
# Define the relu network for comparison 
# For UV-ReLU, UV-ReLU(constrained)
class relu_net_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()    
        self.fc1 = nn.Linear(self.num_input, self.num_hidden_1, bias = False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.num_hidden_1, self.num_output, bias = False)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).squeeze(1)
        return x
    
# For UV
class fc_net_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()    
        self.fc1 = nn.Linear(self.num_input, self.num_hidden_1, bias = False)
        self.fc2 = nn.Linear(self.num_hidden_1, self.num_output, bias = False)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x).squeeze(1)
        return x

#===========================END=============================

# Embed ReLU to UDV
#===========================================================
#==========================START============================
# For UDV-ReLU and UDV-ReLU-s
class UDV_relu_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.num_input, self.num_hidden_1, bias = False) 
        self.diag1 = D_singleConnection(num_in_Diag = self.num_hidden_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.num_hidden_1, self.num_output, bias = False)
           
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.diag1(x)
        x = self.relu(x)
        x = self.fc2(x).squeeze(1)
        return x
#===========================END=============================