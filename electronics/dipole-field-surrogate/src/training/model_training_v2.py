import torch.nn as nn
import torch
import torch.optim as optim
import mlflow
import optuna

def train_fn(train_loader, val_loader, model, epochs=100, learning_rate=1e-3, weight_decay=1e-6, device='cuda', trial=None):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_decay)
    criterion = nn.MSELoss()

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