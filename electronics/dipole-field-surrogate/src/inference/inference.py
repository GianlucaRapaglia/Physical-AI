# %%
import mlflow
import joblib
import torch
import numpy as np
import random

from src.data.data_loaders import get_train_val_loaders
from src.utils.plot import plot_field_vectors
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load model from MLflow
# -----------------------------
model_name = "DipoleFieldModel"
model_uri = f"models:/{model_name}/latest"

model = mlflow.pytorch.load_model(model_uri)
model.eval().to(torch.device("cpu"))

# -----------------------------
# 2. Load scalers from the same run
# -----------------------------
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions(model_name, stages=["None"])[0]
run_id = model_version.run_id

scaler_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="scalers",
)

x_scaler = joblib.load(f"{scaler_path}/x_scaler.save")
y_scaler = joblib.load(f"{scaler_path}/y_scaler.save")

print("Model and scalers loaded successfully!")

# -----------------------------
# 3. Create validation DataLoader
# -----------------------------
_, val_loader, _, _ = get_train_val_loaders(
    batch_size=1,
    grid_size=32,
    grid_range=0.3,
)

# -----------------------------
# 4. Sample a random validation example
# -----------------------------
sample_batch = random.choice(list(val_loader))
sample_input, sample_target = sample_batch

print("Sample input shape:", sample_input.shape)
print("Sample target shape:", sample_target.shape)

# -----------------------------
# 5. Run inference
# -----------------------------
with torch.no_grad():
    predicted_output = model(sample_input)

# -----------------------------
# 6. Inverse scaling
# -----------------------------
pred_flat = predicted_output.numpy().reshape(1, -1)
pred_unscaled = y_scaler.inverse_transform(pred_flat)
pred_unscaled = pred_unscaled.reshape(1, 2, 32, 32)

Ex = pred_unscaled.squeeze()[0]
Ey = pred_unscaled.squeeze()[1]

# -----------------------------
# 7. Recover original input parameters
# -----------------------------
X_unscaled = x_scaler.inverse_transform(sample_input.numpy())
L, x_pos, y_pos, V = X_unscaled.squeeze()

print(f"Input Parameters: L={L}, x_pos={x_pos}, y_pos={y_pos}, V={V}")

# -----------------------------
# 8. Plot field
# -----------------------------
fig, ax = plot_field_vectors(Ex, Ey, L, x_pos, y_pos)
plt.show()
