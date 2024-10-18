# Name: (Regression) UDV training framework part - model 3
# Function: (UDV-s) Model_3, Matrix_uv
# Framework for each method is identical 
#===========================================================
# Necessary package
import torch
from .udvRegVal import reg_valLoop
from .udvConstraints import Matrix_bothside
#===========================================================
#==========================START============================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 3: Matrix method with unconstrained w 
# UDV sturcture with matrix method and unconstrained w in D
# UDV-s
def udv_frame_m_uv(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, constraints_uv):
    model = model.to(device)
    
    # Store train/validation loss
    train_epoch_loss, val_epoch_loss = [], []
        
    # Model training 
    for epoch_index in range (1, num_epochs + 1):
        model, train_loss_buffer = reg_trainLoop_m_uv(model,
                                                      optimizer,
                                                      loss_fn,
                                                      train_loader,
                                                      device,
                                                      constraints_uv
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


# 3: training loop of Matrix_uv
def reg_trainLoop_m_uv(model, optimizer, loss_fn, train_loader, device, constraints_uv):
    model.train()
    train_batch_loss = []
    for train_data, train_targets in train_loader:
        train_data = train_data.to(device)
        train_targets = train_targets.to(device)
        output = model.forward(train_data)
        loss = loss_fn(output, train_targets)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        train_batch_loss.append(loss.item())
        
        # Constraint rule: "update then apply constraints"
        # Note validation loop after this
        with torch.no_grad():  
            model.fc1.weight = constraints_uv(matrix_uv = model.fc1.weight) 
            model.fc2.weight = constraints_uv(matrix_uv = model.fc2.weight)
            
    train_batch_loss = sum(train_batch_loss) / len(train_batch_loss)
   
    return model, train_batch_loss

#===========================END=============================