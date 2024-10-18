# Name: (Regression) UDV training framework part - Val
# Function: validation loop (Shared)
#===========================================================
# Necessary package
import torch
#===========================================================
#==========================START============================
def reg_valLoop(model, loss_fn, val_loader, device):
    model.eval()
    val_batch_loss = []
    for val_data, val_targets in val_loader:
        val_data = val_data.to(device)
        val_targets = val_targets.to(device)
        with torch.no_grad():
            val_output = model.forward(val_data)
            val_loss = loss_fn(val_output, val_targets)
            val_batch_loss.append(val_loss.item())
  
    val_batch_loss = sum(val_batch_loss) / len(val_batch_loss)
    return val_batch_loss
#===========================END=============================