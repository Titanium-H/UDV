# Name: (Regression) UDV training framework part - model 0 
# Function: (UDV-v1) Model_0, Vector_uwv
# Framework for each method is identical 
#===========================================================
# Necessary package
import torch
from .udvRegVal import reg_valLoop
from .udvConstraints import Vector_left_U, UDV_Diag, Vector_right_V
#===========================================================
#==========================START============================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 0: Vector method with constrained w 
# UDV sturcture with vector method and constrained w in D
# UDV-v1
def udv_frame_v_uwv(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, constraints_u, constraints_d, constraints_v):
    model = model.to(device)
    
    # Store train/validation loss
    train_epoch_loss, val_epoch_loss = [], []

    # Model training 
    for epoch_index in range (1, num_epochs + 1):
        model, train_loss_buffer = reg_trainLoop_v_uwv(model,
                                                       optimizer,
                                                       loss_fn,
                                                       train_loader,
                                                       device,
                                                       constraints_u,
                                                       constraints_d,
                                                       constraints_v
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


# 0: training loop of Vector_uwv
def reg_trainLoop_v_uwv(model, optimizer, loss_fn, train_loader, device, constraints_u, constraints_d, constraints_v):
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
            model.fc1.weight = constraints_u(left_u = model.fc1.weight) 
            model.diag1.weight = constraints_d(udv_d = model.diag1.weight)
            model.fc2.weight = constraints_v(right_v = model.fc2.weight)
            
    train_batch_loss = sum(train_batch_loss) / len(train_batch_loss)
    
    return model, train_batch_loss

#===========================END=============================