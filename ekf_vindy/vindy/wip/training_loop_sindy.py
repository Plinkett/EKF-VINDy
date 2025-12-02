# def _training_iteration_sindy(sindy_layer: nn.Module, z_train: torch.Tensor, dzdt_train: torch.Tensor,
#                        l1_weight: float, optimizer: torch.optim.Optimizer):
#         optimizer.zero_grad()

#         # Forward pass (ignoring pruned coefficients!)
#         dzdt_pred = sindy_layer(z_train)
    
#         # Data-fit loss (MSE)
#         mse = F.mse_loss(dzdt_pred, dzdt_train)

#         # L1 sparsity penalty
#         l1 = l1_weight * torch.sum(torch.abs(sindy_layer.big_xi * sindy_layer.mask))
#         loss = mse + l1

#         # Backprop
#         loss.backward()

#         # Update weights
#         optimizer.step()
        
#         return loss, mse, l1

# def train_sindy(
#     sindy_layer: nn.Module,
#     z_train: torch.Tensor, dzdt_train: torch.Tensor,
#     n_epochs = 500,
#     lr = 1e-3,
#     l1_weight = 1e-4,
#     warmup = 50,
#     pruning_epochs = 20,
#     n_epochs_refinement = 1000
# ):
#     """
#     z_train:     (N*T, n_variables)
#     dzdt_train:  (N*T, n_variables)
#     """
    
#     optimizer = torch.optim.SGD([sindy_layer.big_xi], lr=lr)

#     # Zero out gradients of pruned coefficients, such that they don't regrow
#     sindy_layer.big_xi.register_hook(lambda grad: grad * sindy_layer.mask)

#     # Standard training phase with L1 penalty and pruning after warmup
#     for epoch in range(n_epochs):
#         loss, mse, l1 = _training_iteration_sindy(sindy_layer, z_train, dzdt_train, l1_weight, optimizer)

#         # Thresholding
#         if epoch >= warmup and (epoch - warmup) % pruning_epochs == 0:
#             print(f'Pruning at epoch {epoch}')
#             sindy_layer._apply_mask()


#         if epoch % 50 == 0:
#             print(f"[{epoch}] loss={loss.item():.4e}   mse={mse.item():.4e}   L1={l1.item():.4e}")
    
#     sindy_layer._apply_mask()
    
#     print('Refinement phase starting...')
#     # Refinement training phase without pruning nor L1 penalty
#     for epoch in range(n_epochs_refinement):
#         l1_weight = 0.0
#         loss, mse, l1 = _training_iteration_sindy(sindy_layer, z_train, dzdt_train, l1_weight, optimizer)
#         if epoch % 50 == 0:
#             print(f"[{epoch}] loss={loss.item():.4e}   mse={mse.item():.4e}   L1={l1.item():.4e}")
            
#     return sindy_layer