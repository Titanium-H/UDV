# Name: UDV constraints and variations 
# Function: Apply constraints to weights on sandwich strcture
#===========================================================
# Necessary package
import torch
import torch.nn as nn

#===========================================================
#==========================START============================

# Original UDV: Matrix_udv
# U/V bound: the calculation of Left-U/Right-V can be unified by Frobenius Norm
# For UDV, UDV-s (*UDV-ReLU, *UDV-ReLU-s)
class Matrix_bothside(nn.Module):
    def __init__(self, normLim = 1):
        self.normLim = normLim # The norm (upper) limit
        print(f"Matrix_UV: Norm constraints take effect when it > {self.normLim} and normalise to 1")
        super().__init__()
        
    def forward(self, matrix_uv):
        norm_sum = torch.linalg.matrix_norm(matrix_uv, ord = 'fro') ** 2
        if norm_sum > self.normLim:
            matrix_uv = torch.div(matrix_uv, torch.sqrt(norm_sum), out = matrix_uv) 
        return matrix_uv

# D constraints
class UDV_Diag(nn.Module):
    def __init__(self, threshold = 0, boundTo = 0):
        self.threshold = threshold  # Upper limit of weights value
        self.boundTo = boundTo      # Set new value for weights if it greater than upper limit
        print(f"D: Threshold is {self.threshold} and values smaller than it will be bounded to {self.boundTo}")
        super().__init__()
         
    def forward(self, udv_d):
        udv_d[udv_d < self.threshold] = self.boundTo
        return udv_d

#===========================================================

# Vector method
# For UDV-v1 and UDV-v2
# Specifically for 'U' side
class Vector_left_U(nn.Module):
    def __init__(self, normLim = 1):
        self.normLim = normLim # The norm (upper) limit
        print(f"Vector_U: Norm constraints take effect when it > {self.normLim} and normalise to 1")
        super().__init__()
        
    def forward(self, left_u):
        norm_left = torch.linalg.vector_norm(left_u, dim = 1)
        left_u = torch.where(norm_left.unsqueeze(1) > self.normLim, left_u / norm_left.unsqueeze(1), left_u, out = left_u)
        return left_u

# Vector method
# For UDV-v1 and UDV-v2
# Specifically for 'V' side
class Vector_right_V(nn.Module):
    def __init__(self, normLim = 1):
        self.normLim = normLim # The norm (upper) limit 
        print(f"Vector_V: Norm constraints take effect when it > {self.normLim} and normalise to 1")
        super().__init__()
         
    def forward(self, right_v):
        norm_right = torch.linalg.vector_norm(right_v, dim = 0)
        right_v = torch.where(norm_right.unsqueeze(0) > self.normLim, right_v / norm_right.unsqueeze(0), right_v, out = right_v)
        return right_v
    
#===========================END=============================