# Name: Loading pre-processing (Classification)
# Function: load and pre-process dataset(s)
#===========================================================
# Necessary package
import random
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
#===========================================================
#==========================START============================
# Complete pre-processing
def MNIST_Pre(load_All, data_path):
    
    # Pre-processing
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.Grayscale(num_output_channels = 3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
                                   ])
    
    if load_All:
        print("Load all data")
        train_dataset = MNIST(data_path, train = True, download = False, transform = transform)
        val_dataset = MNIST(data_path, train = False, download = False, transform = transform)
        
    else: # Around 1/10 MNIST
        print("Only load part of data for test")
        num_train_samples = 1024 * 6
        mnist_traindataset = MNIST(data_path, train = True, download = False, transform = transform)
        train_indices = random.sample(range(len(mnist_traindataset)), num_train_samples)
        train_dataset = Subset(mnist_traindataset, train_indices)       
        
        num_test_samples = 1024
        mnist_valdataset = MNIST(data_path, train = False, download = False, transform = transform)
        test_indices = random.sample(range(len(mnist_valdataset)), num_test_samples)
        val_dataset = Subset(mnist_valdataset, test_indices)
    
    num_output = 10 # Number of output classes
    
    print(f"{len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    return train_dataset, val_dataset, num_output

#===========================================================
# Simple pre-processing (if necessary)
def MNIST_Pre_Simple(load_All, data_path):
    
    # Pre-processing
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))# Only for MNIST dataset
                                   ])
    
    if load_All:
        print("Load all data")
        train_dataset = MNIST(data_path, train = True, download = False, transform = transform)
        val_dataset = MNIST(data_path, train = False, download = False, transform = transform)
        
    else: # Around 1/10 MNIST
        print("Only load part of data for test")
        num_train_samples = 1024 * 6
        mnist_traindataset = MNIST(data_path, train = True, download = False, transform = transform)
        train_indices = random.sample(range(len(mnist_traindataset)), num_train_samples)
        train_dataset = Subset(mnist_traindataset, train_indices)       
        
        num_test_samples = 1024
        mnist_valdataset = MNIST(data_path, train = False, download = False, transform = transform)
        test_indices = random.sample(range(len(mnist_valdataset)), num_test_samples)
        val_dataset = Subset(mnist_valdataset, test_indices)
    
    num_output = 10 # Number of output classes
    
    print(f"{len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    return train_dataset, val_dataset, num_output

#===========================END=============================