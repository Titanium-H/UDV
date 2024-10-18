# Name: Loss function
# Function: Customised loss function
#===========================================================
# Necessary package
import torch
import torch.nn as nn
#===========================================================
#==========================START============================
# Define the loss function which is half of the mean squared error (squared L2 norm)
# To aviod missing warning, better to directly use MSE (in practice)

class UDV_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_1 = 0
        
    def forward(self, x, y):
        self.loss_1 = torch.mean(torch.pow((x - y), 2)) / 2
        return self.loss_1
    
#===========================END=============================