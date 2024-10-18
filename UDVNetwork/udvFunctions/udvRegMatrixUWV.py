# Name: (Regression) UDV training framework part - model 2
# Function: (UDV) Model_2, Matrix_uwv
# Framework for each method is identical 
#===========================================================
# Necessary package
import torch
from .udvRegVal import reg_valLoop
from .udvConstraints import Matrix_bothside, UDV_Diag
#===========================================================
#==========================START============================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2: Matrix method with constrained w 
# UDV sturcture with matrix method and constrained w in D
# UDV
def udv_frame_m_uwv(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, constraints_uv, constraints_d):
    model = model.to(device)   # Transfer model to device
    
    # Store train/validation loss
    train_epoch_loss, val_epoch_loss = [], []
        
    # Model training 
    for epoch_index in range (1, num_epochs + 1):
        model, train_loss_buffer = reg_trainLoop_m_uwv(model,
                                                       optimizer,
                                                       loss_fn,
                                                       train_loader,
                                                       device,
                                                       constraints_uv,
                                                       constraints_d
                                                      )
        train_epoch_loss.append(train_loss_buffer)
        
        # Model validation
        val_loss_buffer = reg_valLoop(model,
                                      loss_fn,
                                      val_loader,
                                      device
                                     )
        val_epoch_loss.append(val_loss_buffer)
    
        print(f"epoch {epoch_index}/{num_epochs} train_loss: {train_loss_buffer:.12f} val_loss: {val_loss_buffer:.12f}")
        
    # Save weights at the last epoch
    save_weights_list = [model.fc1.weight.clone().detach(),
                         model.diag1.weight.clone().detach(),
                         model.fc2.weight.clone().detach()]
        
    return model, train_epoch_loss, val_epoch_loss, save_weights_list


# 2: training loop of Matrix_uwv
def reg_trainLoop_m_uwv(model, optimizer, loss_fn, train_loader, device, constraints_uv, constraints_d):
    model.train()
    train_batch_loss = []
    for train_data, train_targets in train_loader:
        train_data = train_data.to(device)        # Transfer train data to device
        train_targets = train_targets.to(device)  # Transfer train targets to device
        output = model.forward(train_data)        # Calculate y_hat (model output)
        loss = loss_fn(output, train_targets)     # Calculate the loss
        optimizer.zero_grad()                     # Erase accumulated gradient
        loss.backward()                           # Back propagation
        optimizer.step()                          # Update parameters
        train_batch_loss.append(loss.item())      # Record loss
        
        # Constraint rule: "update then apply constraints"
        # Note validation loop after this
        with torch.no_grad():  
            model.fc1.weight = constraints_uv(matrix_uv = model.fc1.weight) # U constraint
            model.diag1.weight = constraints_d(udv_d = model.diag1.weight)  # D constraint
            model.fc2.weight = constraints_uv(matrix_uv = model.fc2.weight) # V constraibr
            
    train_batch_loss = sum(train_batch_loss) / len(train_batch_loss)        # Average batch losses to epoch loss
   
    return model, train_batch_loss

#===========================END=============================