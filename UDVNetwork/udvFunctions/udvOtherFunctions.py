# Name: Other customised functions
# Function: see comments above of the each function
#===========================================================
# Necessary package
import torch
import io
import pickle
import os
#===========================================================
#==========================START============================
# Other customised functions:

# Use to save loss or other metrics
def store_metrics(a, b):
    if len(a) == 0:
        a = [b]
    else:
        a.append(b)
    return a

# Use to average results among various seeds
def take_avg(list_in):
    if len(list_in) == 0:
        list_out = list_in
    else:    
        sum_list = list_in[0]
        if len(list_in) > 1: 
            for i in range (1, len(list_in)):
                sum_list = list(map(lambda x, y: x + y, sum_list, list_in[i]))
        list_out = [x / len(list_in) for x in sum_list]
    
    return list_out

# Check tensor shape before assigning weight matrices to neurons
def check_shapes(model_weight, list_sample):
    if model_weight.shape != list_sample.shape:
        raise ValueError("Check shapes!")
    return None

# Averge results from last few epochs
def avg_stable(orig_list, last_n):
    if len(orig_list) < last_n:
        raise ValueError("List should contain at least {0} numbers".format(last_n))
    average = sum(orig_list[-last_n:]) / last_n
    return average

# Check non-value (Tensor or value)
def non_Value(value):
    if isinstance(value, torch.Tensor):  
        return torch.isnan(value).any() or torch.isinf(value).any()
    else:
        return value != value or value in (float('inf'), float('-inf'))

# Create a list contains number of neurons for SVD-Pruning (Classification)
# Note the baseline test should NOT be in the same loop (start_number NOT in the list)
def pruning_list(start_number, deduce_factor):
    number_neurons_list = []
    while int(start_number * deduce_factor) >= 1:
        start_number *= deduce_factor
        int_number = int(start_number)
        if int_number not in number_neurons_list:
            number_neurons_list.append(int_number)
    return number_neurons_list

# Compare the saved weights from pt file and pickle file
# (Not necessary)
def same_saving(u1, w1, v1, u2, w2, v2):
    if not (torch.all(u1 == u2) and torch.all(w1 == w2) and torch.all(v1 == v2)):
        raise ValueError("Weights from saved file and saved model are NOT same")
    return None

# Get folders in the current directory
def obtain_current_directory(endString):
    current_directory = os.getcwd()
    read_file_list = [f.name for f in os.scandir(current_directory) if f.is_dir() and f.name.endswith(endString)]
    folder_list = sorted(read_file_list, reverse = True)

    for subfolder in folder_list:
        print(subfolder)
    print("The number of foders is: ", len(folder_list))
    return folder_list

# Read saved file (parameters, metrics, etc.)
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
#===========================END=============================