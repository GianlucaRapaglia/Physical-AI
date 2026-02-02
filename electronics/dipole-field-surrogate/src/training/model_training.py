from src.utils.paths import PROCESSED_DATA_DIR
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.model import DipoleFieldModel
import torch.optim as optim
import mlflow
import mlflow.pytorch
import optuna

def train_fn(train_loader, val_loader, latent_channels=64, latent_h=8, latent_w=8, batch_size=32, epochs=100, learning_rate=1e-3, weight_decay=1e-6, device='cuda', trial=None):

    model = DipoleFieldModel(
        input_dim=train_loader.dataset[0][0].shape[0],
        latent_channels=latent_channels,
        latent_h=latent_h,
        latent_w=latent_w,
        output_channels=train_loader.dataset[0][1].shape[0]
    ).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Logging in MLFlow
    mlflow.log_param("latent_channels", latent_channels)
    mlflow.log_param("latent_h", latent_h)
    mlflow.log_param("latent_w", latent_w)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_dict(
        {
            "input_features": "L, x_pos, y_pos, V",
            "output_features": "Ex, Ey"
        },
        "features.json"
    )

    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for inputs_batch, targets_batch in train_loader:
            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()

            outputs_batch = model(inputs_batch)

            loss = criterion(outputs_batch, targets_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs_batch, targets_batch in val_loader:

                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)

                outputs_batch = model(inputs_batch)


                assert outputs_batch.shape == targets_batch.shape, f"Validation output and target must have the same shape"

                loss = criterion(outputs_batch, targets_batch)
                running_val_loss += loss.item() * inputs_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
 
        if trial is not None:
            trial.report(epoch_val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()   

    return epoch_val_loss # Return the final validation loss