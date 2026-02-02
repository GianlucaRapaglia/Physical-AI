import optuna 
import mlflow
import numpy as np
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils.paths import PROCESSED_DATA_DIR, MLFLOW_TRACKING_URI
from src.models.model import DipoleFieldModel

from src.training.model_training import train_fn

from src.data.data_loaders import get_train_val_loaders

BATCH_SIZE = 32

train_loader, val_loader, X_scaler, y_scaler = get_train_val_loaders(
    batch_size=BATCH_SIZE,
    grid_size=32,
    grid_range=0.3,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
tracking_uri = mlflow.get_tracking_uri()
print(f"Current tracking uri: {tracking_uri}")

mlflow.set_experiment("DipoleFieldModel")

def objective(trial):
    latent_channels = trial.suggest_categorical("latent_channels", [32, 64, 128])
    latent_h = trial.suggest_categorical("latent_h", [4, 8, 16])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    epochs = trial.suggest_categorical("epochs", [50, 100, 200])

    # Train the model with the suggested hyperparameters
    with mlflow.start_run():

        val_loss = train_fn(
            train_loader,
            val_loader,
            latent_channels=latent_channels,
            latent_h=latent_h,
            latent_w=latent_h,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            learning_rate=learning_rate,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            trial=trial
        )

    # Log the validation loss to Optuna
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2)

# Save best model, hyperparameters, and scalers
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Save the best model using MLflow
final_model = DipoleFieldModel(
    input_dim=train_loader.dataset[0][0].shape[0],
    latent_channels=best_trial.params["latent_channels"],
    output_channels=train_loader.dataset[0][1].shape[0]).to(device='cuda' if torch.cuda.is_available() else 'cpu')

# Training the final model on the entire training set
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_trial.params["learning_rate"])
final_model.train()

for _ in range(best_trial.params["epochs"]):
    for inputs_batch, targets_batch in train_loader:

        inputs_batch = inputs_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        targets_batch = targets_batch.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer.zero_grad()

        outputs_batch = final_model(inputs_batch)

        loss = torch.nn.MSELoss()(outputs_batch, targets_batch)
        loss.backward()

        optimizer.step()

# Log the final model with MLflow
with mlflow.start_run(run_name="Best_Model"):
    mlflow.pytorch.log_model(
        final_model,
        artifact_path="model",
        registered_model_name="DipoleFieldModel"
    )
    for key, value in best_trial.params.items():
        mlflow.log_param(key, value)

    # Log scalers
    mlflow.log_artifact(PROCESSED_DATA_DIR / "x_scaler.save", artifact_path="scalers")
    mlflow.log_artifact(PROCESSED_DATA_DIR / "y_scaler.save", artifact_path="scalers")

    # Log data path
    # mlflow.log_artifact(str(PROCESSED_DATA_DIR / "circuit_data_preprocessed.npz"), artifact_path="data")


