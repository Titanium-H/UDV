# Name: Diagonal Layer
# Function: In practice, single connection layer
#===========================================================
# Necessary package
import torch
import torch.nn as nn
#===========================================================
#==========================START============================
# Define the single-connection layer that carry weight matrix 'w'
# "Diagonal Layer"

class D_singleConnection(nn.Module):
    def __init__(self,num_in_Diag):
        super().__init__() 
        
        # Define single connection (weight will be overwritten by reproducible experiment)
        self.num_in_Diag = num_in_Diag
        self.weight = nn.Parameter(torch.randn(1, self.num_in_Diag))
        
    def forward(self, x):
        return torch.mul(self.weight, x)
    
#===========================END=============================