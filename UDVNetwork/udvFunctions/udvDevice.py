# Name: Load device
# Function: GPU/CPU/MPS
#===========================================================
# Necessary package
import torch
#===========================================================
#==========================START============================
# Check the GPU/CPU

def get_device():
    """Get the device to be used for computations."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # MPS does not support float64
    else:
        device = "cpu"
    print(f"Using {device} device")
    return device
#===========================END=============================