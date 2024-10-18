# Name: UDV identical initialisation 
# Function: Identical initialisation for all models
#===========================================================
# Necessary package
import torch
import torch.nn as nn

#===========================================================
#==========================START============================
# Re-producible setting: Use public seed to generate parameters matrices 
# To ensure the same initialization on different methods

def seedList1(num_seeds, public_seed, num_input, num_hidden_1, num_output):
    init_ulist1 = []     # Generate weight matrices for the hidden layer 1
    init_wlist1 = []     # Generate weight vector for constrained layer
    init_vlist1 = []     # Generate weight matrices for the output layer (hidden layer 2)
    
    torch.manual_seed(public_seed)
    
    # Same initialization method as the pytorch default way
    for o in range (0, num_seeds):
        init_u_1 = torch.empty(num_hidden_1, num_input)
        init_w_1 = torch.empty(1, num_hidden_1)
        init_v_1 = torch.empty(num_output, num_hidden_1)
        init_u_1 = nn.init.uniform_(init_u_1, a = -((1/num_input)**0.5), b = ((1/num_input)**0.5))
        init_w_1 = nn.init.uniform_(init_w_1, a = -((1/num_hidden_1)**0.5), b = ((1/num_hidden_1)**0.5))
        init_v_1 = nn.init.uniform_(init_v_1, a = -((1/num_hidden_1)**0.5), b = ((1/num_hidden_1)**0.5))
        init_ulist1.append(init_u_1)
        init_wlist1.append(init_w_1)
        init_vlist1.append(init_v_1)
        
    return init_ulist1, init_wlist1, init_vlist1

#===========================END=============================