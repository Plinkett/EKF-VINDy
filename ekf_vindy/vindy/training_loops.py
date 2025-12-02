import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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