# %%
import optuna 
import mlflow
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils.paths import PROCESSED_DATA_DIR, MLFLOW_TRACKING_URI
from src.models.model_v2 import SurrogateModel

from src.training.model_training_v2 import train_fn

from src.data.data_loaders_v2 import get_train_val_loaders

BATCH_SIZE = 32
SEED = 42

# Setting MLFlow Tracking URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
tracking_uri = mlflow.get_tracking_uri()
print(f"Current tracking uri: {tracking_uri}")

# Setting device to be used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Creating the dataloaders

train_loader, val_loader, X_scaler, y_scaler = get_train_val_loaders(
    batch_size=BATCH_SIZE,
    grid_size=32,
    grid_range=0.3,
    seed=SEED,
)

input_dim = train_loader.dataset[0][0].shape[0]
output_dim = train_loader.dataset[0][1].shape[0]

print(f"Input dim: {input_dim}, Output dim: {output_dim}")

mlflow.set_experiment("DipoleFieldModel_Spatial")

# Defining optimization problem for Optuna

def objective(trial):
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    epochs = trial.suggest_categorical("epochs", [50, 100])

    model = SurrogateModel(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        input_dim=input_dim,
        output_dim=output_dim
    ).to(device=device)

    # Train the model with the suggested hyperparameters
    with mlflow.start_run():

        # Logging in MLFlow
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_layers", n_layers)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_dict(
            {
                "input_features": "L, x_pos, y_pos, x, y, V",
                "output_features": "Ex, Ey"
            },
            "features.json"
        )

        val_loss = train_fn(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            epochs=epochs,
            device=device,
            trial=trial
        )

    # Log the validation loss to Optuna
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Save best model, hyperparameters, and scalers
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Save the best model using MLflow
final_model = SurrogateModel(
    input_dim=input_dim,
    output_dim=output_dim,
    n_layers=best_trial.params["n_layers"],
    hidden_dim=best_trial.params["hidden_dim"],
    ).to(device=device)

final_model.train()

# Train the final model with the best hyperparameters
with mlflow.start_run(run_name="Best_model"):

    # Logging in MLFlow

    mlflow.pytorch.log_model(
        final_model,
        artifact_path="model",
        registered_model_name="DipoleFieldModel_Spatial"
    )

    for key, value in best_trial.params.items():
        mlflow.log_param(key, value)

    # Log scalers
    mlflow.log_artifact(PROCESSED_DATA_DIR / "x_scaler.save", artifact_path="scalers")
    mlflow.log_artifact(PROCESSED_DATA_DIR / "y_scaler.save", artifact_path="scalers")

    mlflow.log_param("seed", SEED)
    mlflow.log_param("n_layers", best_trial.params['n_layers'])
    mlflow.log_param("hidden_dim", best_trial.params['hidden_dim'])
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", best_trial.params['epochs'] )
    mlflow.log_param("learning_rate", best_trial.params['learning_rate'])
    mlflow.log_dict(
        {
            "input_features": "L, x_pos, y_pos, x, y, V",
            "output_features": "Ex, Ey"
        },
        "features.json"
    )

    val_loss = train_fn(
        train_loader=train_loader,
        val_loader=val_loader,
        model=final_model,
        epochs=best_trial.params['epochs'],
        learning_rate=best_trial.params['learning_rate'],
        device=device
    )

# %%
torch.save(final_model.state_dict(), 'src/sweeps/models/dipole_surrogate_model.pth')
# Saving the scalers
import joblib
joblib.dump(X_scaler, 'src/sweeps/models/x_scaler.save')
joblib.dump(y_scaler, 'src/sweeps/models/y_scaler.save')


