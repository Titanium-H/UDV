# Name: (Classification) UDV training framework part - Val
# Function: validation loop (Shared)
#===========================================================
# Necessary package
import torch
#===========================================================
#==========================START============================
def class_valLoop(model, loss_fn, val_loader, device):
    model.eval()
    val_batch_loss = 0
    val_batch_acc = 0
    
    for val_data, val_targets in val_loader:
        val_data = val_data.to(device)
        val_targets = val_targets.to(device)
        
        with torch.no_grad():  
            val_output = model.forward(val_data)
            val_loss = loss_fn(val_output, val_targets)
            val_batch_loss += val_loss.item()
            val_batch_acc += (val_output.argmax(1) == val_targets).float().sum().item()
    
    val_batch_loss /= len(val_loader)
    val_batch_acc /= len(val_loader.dataset)
    
    return val_batch_loss, val_batch_acc
#===========================END=============================