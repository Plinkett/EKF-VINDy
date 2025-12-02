import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def _training_iteration_sindy(sindy_layer: nn.Module, z_train: torch.Tensor, dzdt_train: torch.Tensor,
                       l1_weight: float, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()

        # Forward pass (ignoring pruned coefficients!)
        dzdt_pred = sindy_layer(z_train)
    
        # Data-fit loss (MSE)
        mse = F.mse_loss(dzdt_pred, dzdt_train)

        # L1 sparsity penalty
        l1 = l1_weight * torch.sum(torch.abs(sindy_layer.big_xi * sindy_layer.mask))
        loss = mse + l1

        # Backprop
        loss.backward()

        # Update weights
        optimizer.step()
        
        return loss, mse, l1

def train_sindy(
    sindy_layer: nn.Module,
    z_train: torch.Tensor, dzdt_train: torch.Tensor,
    n_epochs = 500,
    lr = 1e-3,
    l1_weight = 1e-4,
    warmup = 50,
    pruning_epochs = 20,
    n_epochs_refinement = 1000
):
    """
    z_train:     (N*T, n_variables)
    dzdt_train:  (N*T, n_variables)
    """
    
    optimizer = torch.optim.SGD([sindy_layer.big_xi], lr=lr)

    # Zero out gradients of pruned coefficients, such that they don't regrow
    sindy_layer.big_xi.register_hook(lambda grad: grad * sindy_layer.mask)

    # Standard training phase with L1 penalty and pruning after warmup
    for epoch in range(n_epochs):
        loss, mse, l1 = _training_iteration_sindy(sindy_layer, z_train, dzdt_train, l1_weight, optimizer)

        # Thresholding
        if epoch >= warmup and (epoch - warmup) % pruning_epochs == 0:
            print(f'Pruning at epoch {epoch}')
            sindy_layer._apply_mask()


        if epoch % 50 == 0:
            print(f"[{epoch}] loss={loss.item():.4e}   mse={mse.item():.4e}   L1={l1.item():.4e}")
    
    sindy_layer._apply_mask()
    
    print('Refinement phase starting...')
    # Refinement training phase without pruning nor L1 penalty
    for epoch in range(n_epochs_refinement):
        l1_weight = 0.0
        loss, mse, l1 = _training_iteration_sindy(sindy_layer, z_train, dzdt_train, l1_weight, optimizer)
        if epoch % 50 == 0:
            print(f"[{epoch}] loss={loss.item():.4e}   mse={mse.item():.4e}   L1={l1.item():.4e}")
            
    return sindy_layer

def train_vindy(
    vindy_layer: nn.Module,
    z_train: torch.Tensor,
    dzdt_train: torch.Tensor,
    n_epochs = 1000,
    batch_size = 256,
    lr = 1e-3,
    huber_weight = 1.0,
    kl_weight = 1e-3,
):

    # Dataset + DataLoader
    dataset = TensorDataset(z_train, dzdt_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(vindy_layer.parameters(), lr=lr)

    for epoch in range(n_epochs):

        for batch_idx, (z_batch, dzdt_batch) in enumerate(loader):

            optimizer.zero_grad()

            # Forward pass on minibatch
            dzdt_pred = vindy_layer(z_batch, sample = True)

            # Huber loss
            huber = F.huber_loss(dzdt_pred, dzdt_batch, delta=1.0, reduction='mean')

            # KL term. Notice that this scales with the size of the library!
            kl = vindy_layer.big_xi_distribution.kl_divergence(
                vindy_layer.laplace_prior
            ).sum()

            # Total objective
            loss = huber_weight * huber + kl_weight * kl
        
            # Backprop + update
            loss.backward()
            optimizer.step()
            
            # Print loss per batch
            print(f"[Epoch {epoch} | Batch {batch_idx}] "
                  f"loss={loss.item():.4e} huber={huber.item():.4e} KL={kl.item():.4e}")
            
    return vindy_layer