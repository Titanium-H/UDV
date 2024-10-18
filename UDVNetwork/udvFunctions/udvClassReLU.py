# Name: (Classification) UDV training framework part - model 4
# Function: (UV-ReLU) Model_4, ReLU
# Subfunction: UDV-ReLU, UDV-ReLU-s
#===========================================================
# Necessary package
import torch
from .udvClassVal import class_valLoop
#===========================================================
#==========================START============================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 4: ReLU
# Unconstrained networks
# UV-ReLU
def udv_frame_relu(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device):
    model = model.to(device)
    
    # Store train/validation loss
    train_epoch_loss, val_epoch_loss = [], []
    train_epoch_acc, val_epoch_acc = [], []

    # Model training 
    for epoch_index in range (1, num_epochs + 1):
        model, train_loss_buffer, train_acc_buffer = class_trainLoop_relu(model,
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
    
    # Save weights at the last epoch
    save_weights_list = [model.classifier.fc1.weight.clone().detach(),
                         model.classifier.fc2.weight.clone().detach()]
    
    return model, train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc, save_weights_list


# 4: training loop of ReLU
def class_trainLoop_relu(model, optimizer, loss_fn, train_loader, device):
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


# Name: (Classification) UDV training framework (Embed ReLU to UDV)
# Function: UDV-ReLU, UDV-ReLU-s
#===========================================================
from .udvConstraints import Matrix_bothside
#===========================================================
#==========================START============================

# UDV sturcture with matrix method and unconstrained w in D in relu
# UDV-ReLU, UDV-ReLU-s
def udvrelu_frame_m_uv(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, constraints_uv):
    model = model.to(device)
    
    # Store train/validation loss
    train_epoch_loss, val_epoch_loss = [], []
    train_epoch_acc, val_epoch_acc = [], []

    # Model training 
    for epoch_index in range (1, num_epochs + 1):
        model, train_loss_buffer, train_acc_buffer = class_trainLoop_m_uv_relu(model,
                                                                               optimizer,
                                                                               loss_fn,
                                                                               train_loader,
                                                                               device,
                                                                               constraints_uv
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
    
    # Save weights at the last epoch
    save_weights_list = [model.classifier.fc1.weight.clone().detach(),
                         model.classifier.fc2.weight.clone().detach()]
    
    return model, train_epoch_loss, val_epoch_loss, train_epoch_acc, val_epoch_acc, save_weights_list


# 3: training loop of Matrix_uv
def class_trainLoop_m_uv_relu(model, optimizer, loss_fn, train_loader, device, constraints_uv):
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
        
        # Constraint rule: "update then apply constraints"
        # Note validation loop after this
        with torch.no_grad():
            model.classifier.fc1.weight = constraints_uv(matrix_uv = model.classifier.fc1.weight)
            model.classifier.fc2.weight = constraints_uv(matrix_uv = model.classifier.fc2.weight)
            
    train_batch_loss /= len(train_loader)
    train_batch_acc /= len(train_loader.dataset)
    return model, train_batch_loss, train_batch_acc


#===========================END=============================