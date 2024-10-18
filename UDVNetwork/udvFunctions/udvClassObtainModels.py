# Name: (Classification) Obtain transfer learning based model 
# Function: Get the originial transferred model and revise the top layers
# Subfunction: Pruning
# *Subfunctions: Simple NN structure for classification
# Subfunction: UDV-ReLU
#===========================================================
# Necessary package
from torchvision import models
import torch
import torch.nn as nn
from collections import OrderedDict
from udvFunctions.udvDiagonalLayer import D_singleConnection
from torchvision.models.regnet import RegNet
#===========================================================
#==========================START============================

# Step 1: Get classifier mapping from original model
def model_InitPara(name, scale_factor):
    if name == 'efficientnet_b0':
        model_classifier = models.efficientnet_b0(weights = 'DEFAULT').classifier
        num_input = models.efficientnet_b0(weights = 'DEFAULT').classifier[1].in_features
        batch_size = 384
    elif name == 'maxvit_t':
        model_classifier = models.maxvit_t(weights = 'DEFAULT').classifier
        num_input = models.maxvit_t(weights = 'DEFAULT').classifier[3].in_features
        batch_size = 128
    elif name == 'regnet_x_32gf':
        model_classifier = models.regnet_x_32gf(weights = 'DEFAULT').fc
        num_input = models.regnet_x_32gf(weights = 'DEFAULT').fc.in_features
        batch_size = 128
    else:
        raise ValueError("Wrong model selection")
    
    num_hidden_1 = round(num_input * scale_factor)
    print(f"current model classifier is {name} which is \n{model_classifier}\n\
Current batch size is {batch_size}")
    
    return num_input, num_hidden_1, batch_size

# Step 2: Re-define the classifier 
def revised_model(name, train_features, pre_trained, model_order, num_input, num_hidden_1, num_output):
    
    # Set pre trained weights
    if pre_trained == False:
        pre_trained = None
    else:
        pre_trained = 'DEFAULT'
        
    # Model: efficientnet_b0
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights = pre_trained)
        if model_order < 4: # UDV-v1, UDV-v2, UDV, UDV-s
            head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 4: # UV-ReLU
            head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 5: # UV
            head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        else:
            head = model.classifier
        
    # Model: maxvit_t    
    elif name == 'maxvit_t':
        model = models.maxvit_t(weights = pre_trained)
        if model_order < 4:
            head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                              ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                              ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 4:
            head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                              ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                              ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 5:
            head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                              ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                              ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        else:
            head = model.classifier
            
    # Model: RegNetX-32GF        
    elif name == 'regnet_x_32gf':
        model = ReTop_regnet(models.regnet_x_32gf(weights = pre_trained))
        
        if model_order < 4:
            head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 4:
            head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 5: # Linear activation
            head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        else:
            head = model.classifier
        
    else:
        raise ValueError("Wrong model selection")
    
    # Rebuild the classifier
    model.classifier = head
    
    # Set trainable layers
    for param in model.parameters():
        param.requires_grad = train_features
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    print(f"Current model is: {name}\n\
Feature layers trainable? {train_features}\n\
Pre-trained weights? {pre_trained} (None = False; 'DEFAULT' = True)\n\
The classifier (top layers) as follows:\n{model.classifier}")
    
    return model


#==========Pre-process the model RegNetX-32GF==============
# RegNet use 'fc' layer as classifier without structing it as "classifier", to reduce code workload, we frist structre the "classifier" by originial layers
# Now the 'fc' layer has the name "classifier"
class ReTop_regnet(nn.Module):
    def __init__(self, original_model):
        super(ReTop_regnet, self).__init__()
        self.stem = original_model.stem
        self.trunk_output = original_model.trunk_output
        self.avgpool = original_model.avgpool
        num_features = original_model.fc.in_features
        self.classifier = nn.Sequential(nn.Linear(num_features, 1000))

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim = 1)
        x = self.classifier(x)
        return x
    
    
#===========================END=============================
# Subfunction: Pruning
# After load the transferred model, this function will load the trained parameters and rebuild the classifier for pruning experiments
def pruning_model(model, name, num_input, num_hidden_new, num_output, ptPath, device):
    
    # Load saved model and set all layers NOT trainable
    model.load_state_dict(torch.load(ptPath))
    for param in model.parameters():
        param.requires_grad = False
    
    # Re-Construct the model top layers (classifier)
    if name == 'efficientnet_b0':
        head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                          ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_new, bias = False)),
                                          ('diag1', D_singleConnection(num_in_Diag = num_hidden_new)),
                                          ('fc2', nn.Linear(in_features = num_hidden_new, out_features = num_output, bias = False))
                                         ]))
        
    elif name == 'maxvit_t':
        head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                          ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                          ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                          ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_new, bias = False)),
                                          ('diag1', D_singleConnection(num_in_Diag = num_hidden_new)),
                                          ('fc2', nn.Linear(in_features = num_hidden_new, out_features = num_output, bias = False))
                                         ]))
        
    elif name == 'regnet_x_32gf':
        head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_new, bias = False)),
                                          ('diag1', D_singleConnection(num_in_Diag = num_hidden_new)),
                                          ('fc2', nn.Linear(in_features = num_hidden_new, out_features = num_output, bias = False))
                                         ]))
    
    else:
        raise ValueError("Wrong model selection")
    
    # Replace the top layers
    model.classifier = head
    
    print("Model and top layers have been overwritten\n")
    
    model.to(device)
    return model
 
#===========================END=============================


# Name: Classification with simple neural network structure
# *Subfunction: Define simple neural netwrok
#===========================================================
#==========================START============================
# Structure for UDV, UDV-s, UDV-v1, UDV,v2
class UDV_net_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()    
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.num_input, self.num_hidden_1, bias = False)),
                ('diag1', D_singleConnection(num_in_Diag = self.num_hidden_1)),
                ('fc2', nn.Linear(self.num_hidden_1, self.num_output, bias = False))
            ])
        )
           
    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
# Structure for UV-ReLU
class relu_net_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()    
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.num_input, self.num_hidden_1, bias = False)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(self.num_hidden_1, self.num_output, bias = False))
            ])
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
# Structure for UV
class fc_net_1(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_output):
        super().__init__()
        self.num_input = num_input
        self.num_hidden_1 = num_hidden_1
        self.num_output = num_output
        
        self.flatten = nn.Flatten()    
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(self.num_input, self.num_hidden_1, bias = False)),
                ('fc2', nn.Linear(self.num_hidden_1, self.num_output, bias = False))
            ])
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x
#===========================END=============================



# Name: UDV-ReLU
# Function: Embed ReLU to UDV structure (Transfer learning)
#===========================================================
#==========================START============================

def revised_model_udvrelu(name, train_features, pre_trained, model_order, num_input, num_hidden_1, num_output):
    
    # Set pre trained weights
    if pre_trained == False:
        pre_trained = None
    else:
        pre_trained = 'DEFAULT'
        
    # Model: efficientnet_b0
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights = pre_trained)
        if model_order == 0 or model_order == 1: # UDV, UDV-s
            head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
        elif model_order == 2 or model_order == 3: # UDV-ReLU, UDV-ReLU-s
            head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
        
        elif model_order == 4 or model_order == 5: # UV-ReLU, UV-ReLU(constrained)
            head = nn.Sequential(OrderedDict([('dp1', nn.Dropout(p = 0.2, inplace = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        else:
            head = model.classifier
        
    # Model: maxvit_t    
    elif name == 'maxvit_t':
        model = models.maxvit_t(weights = pre_trained)
        if model_order == 0 or model_order == 1:
            head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                              ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                              ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 2 or model_order == 3:
            head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                              ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                              ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 4 or model_order == 5:
            head = nn.Sequential(OrderedDict([('ap1', nn.AdaptiveAvgPool2d(output_size = 1)),
                                              ('fl1', nn.Flatten(start_dim = 1, end_dim = -1)),
                                              ('ln1', nn.LayerNorm((512,), eps = 1e-05, elementwise_affine = True)),
                                              ('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        else:
            head = model.classifier
            
            
    elif name == 'regnet_x_32gf':
        model = ReTop_regnet(models.regnet_x_32gf(weights = pre_trained))
        
        if model_order == 0 or model_order == 1:
            head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 2 or model_order == 3:
            head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('diag1', D_singleConnection(num_in_Diag = num_hidden_1)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        elif model_order == 4 or model_order == 5:
            head = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features = num_input, out_features = num_hidden_1, bias = False)),
                                              ('relu', nn.ReLU()),
                                              ('fc2', nn.Linear(in_features = num_hidden_1, out_features = num_output, bias = False))
                                             ]))
            
        else:
            head = model.classifier
        
    else:
        raise ValueError("Wrong model selection")
    
    # Rebuild the classifier
    model.classifier = head
    
    # Set trainable layers
    for param in model.parameters():
        param.requires_grad = train_features
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    print(f"Current model is: {name}\n\
Feature layers trainable? {train_features}\n\
Pre-trained weights? {pre_trained} (None = False; 'DEFAULT' = True)\n\
The classifier (top layers) as follows:\n{model.classifier}")
    
    return model

#===========================END=============================