# Name: (Classification) UDV training framework part - model 6
# Function: (M,E,R) Model_6, Original transfer learning model
# Framework for each method is identical 
#===========================================================
# Necessary package
import torch
from .udvClassVal import class_valLoop
#===========================================================
#==========================START============================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 6: Original transfer learning model
# Unconstrained networks
# For original transferred model: MaxVit-T, EfficientNet-B0, RegNetX-32GF
def udv_frame_Orig(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device):
    model = model.to(device)
    
    # Store train/validation loss
    train_epoch_loss, val_epoch_loss = [], []
    train_epoch_acc, val_epoch_acc = [], []

    # Model training 
    for epoch_index in range (1, num_epochs + 1):
        model, train_loss_buffer, train_acc_buffer = class_trainLoop_Orig(model,
                                                                          optimizer,
                                                                          loss_fn,
                                                                          train_loader,
                                                                          device
                                                                         )
        train_epoch_loss.append(train_loss_buffer)
        train_epoch_acc.append(train_acc_buffer)
        
        # Model validation
        val_loss_buffer, val_acc_buffer = class_valLoop(model,
                                                        loss_fn,
                                                        val_loader,
                                                        device
                                                       )
        val_epoch_loss.append(val_loss_buffer)
        val_epoch_acc.append(val_acc_buffer)
        
        print(f"epoch {epoch_index}/{num_epochs} train_loss: {train_loss_buffer:.12f} val_loss: {val_loss_buffer:.12f} train_acc: {train_acc_buffer:.6f} val_acc: {val_acc_buffer:.6f} ")
    
    return model, train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc


# 6: training loop of original transfer learning model
def class_trainLoop_Orig(model, optimizer, loss_fn, train_loader, device):
    model.train()
    train_batch_loss = 0
    train_batch_acc = 0
    
    for train_data, train_targets in train_loader:
        train_data = train_data.to(device)
        train_targets = train_targets.to(device)
        train_output = model.forward(train_data)
        train_loss = loss_fn(train_output, train_targets)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_batch_loss += train_loss.item()
        train_batch_acc += (train_output.argmax(1) == train_targets).float().sum().item()

    train_batch_loss /= len(train_loader)
    train_batch_acc /= len(train_loader.dataset)
    return model, train_batch_loss, train_batch_acc

#===========================END=============================