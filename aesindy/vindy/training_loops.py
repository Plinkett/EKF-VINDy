import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: have a refining phase. Once it has found 
def train_sindy(
    sindy_layer: nn.Module,
    z_train: torch.Tensor, dzdt_train: torch.Tensor,
    n_epochs = 500,
    lr = 1e-3,
    l1_weight = 1e-4,
    warmup = 50,
    pruning_epochs = 20
):
    """
    z_train:     (N*T, n_variables)
    dzdt_train:  (N*T, n_variables)
    """
    
    optimizer = torch.optim.Adam([sindy_layer.big_xi], lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass (ignoring pruned coefficients!)
        big_xi_masked = sindy_layer.big_xi * sindy_layer.mask
        dzdt_pred = sindy_layer._evaluate_theta(z_train) @ big_xi_masked

        # Data-fit loss (MSE)
        mse = F.mse_loss(dzdt_pred, dzdt_train)

        # L1 sparsity penalty
        l1 = l1_weight * torch.sum(torch.abs(big_xi_masked))
        loss = mse + l1

        # Backprop
        loss.backward()

        # Gradient step
        optimizer.step()

        # Thresholding
        if epoch >= warmup and (epoch - warmup) % pruning_epochs == 0:
            print(f'Pruning at epoch {epoch}')
            sindy_layer._apply_mask()


        if epoch % 50 == 0:
            print(f"[{epoch}] loss={loss.item():.4e}   mse={mse.item():.4e}   L1={l1.item():.4e}")
    
    sindy_layer._apply_mask()
    return sindy_layer