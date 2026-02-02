# %%
import mlflow
import joblib
import torch
import numpy as np
import random
from scipy.stats import qmc
import torch.nn as nn
import matplotlib.pyplot as plt
from src.utils.plot import plot_field_vectors

def plot_field_vectors(Ex, Ey, grid_x=None, grid_y=None,
                       x_pos=None, y_pos=None, L=None,
                       skip=4, cmap='hot', title='Electric Field Vectors (Normalized)'):
    """
    Plot normalized electric field vectors (quiver), colored by magnitude.
    Returns (fig, ax).
    """
    E_mag = np.sqrt(Ex**2 + Ey**2)
    Ex_norm = Ex / (E_mag + 1e-10)
    Ey_norm = Ey / (E_mag + 1e-10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip],
              Ex_norm[::skip, ::skip], Ey_norm[::skip, ::skip],
              E_mag[::skip, ::skip],
              cmap=cmap,
              scale=None,
              scale_units='xy',
              angles='xy',
              width=0.004,
              headwidth=4,
              headlength=5)

    ax.plot([x_pos, x_pos], [y_pos - L/2, y_pos + L/2], 'b-', linewidth=3)
    ax.plot(x_pos, y_pos + L/2, 'ro', markersize=10, label='+Q', zorder=10)
    ax.plot(x_pos, y_pos - L/2, 'bo', markersize=10, label='-Q', zorder=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax

def compute_dipole_field(L, x_pos, y_pos, V, grid_x, grid_y):
    """
    Compute dipole electric field on a 2D grid
    
    Parameters:
    -----------
    L : float
        Antenna length (meters)
    x_pos : float
        X-position of antenna center
    y_pos : float
        Y-position of antenna center
    V : float
        Applied voltage (volts)
    grid_x : ndarray (2D)
        X-coordinates of grid points (from meshgrid)
    grid_y : ndarray (2D)
        Y-coordinates of grid points (from meshgrid)
    
    Returns:
    --------
    Ex_total : ndarray (2D)
        X-component of electric field at each grid point
    Ey_total : ndarray (2D)
        Y-component of electric field at each grid point
    """

    # 1. CHARGE MAGNITUDE
    # Simplified model: Q ∝ V × L
    Q = V * L * 1e-11  # Changed from 1e-9 to get more reasonable field values
    
    # 2. CHARGE POSITIONS
    x_plus = x_pos         # X-position of positive charge
    y_plus = y_pos + L/2   # Y-position of positive charge (top)
    
    x_minus = x_pos        # X-position of negative charge
    y_minus = y_pos - L/2  # Y-position of negative charge (bottom)
    
    # 3. COULOMB'S CONSTANT
    k = 8.99e9  # More accurate value: N⋅m²/C²
    
    # 4. FIELD FROM POSITIVE CHARGE (+Q)
    # Vector from charge to grid points
    dx_plus = grid_x - x_plus  # Shape: (grid_size, grid_size)
    dy_plus = grid_y - y_plus
    
    # Distance from positive charge to each grid point
    r_plus = np.sqrt(dx_plus**2 + dy_plus**2)
    
    # Avoid division by zero (minimum distance threshold)
    r_plus = np.maximum(r_plus, 1e-6)  # Clamp to minimum value
    
    # Electric field components from positive charge
    # E = k*Q*(r - r_charge)/|r - r_charge|³
    Ex_plus = k * Q * dx_plus / (r_plus**3)
    Ey_plus = k * Q * dy_plus / (r_plus**3)
    
    # 5. FIELD FROM NEGATIVE CHARGE (-Q)
    # Vector from charge to grid points
    dx_minus = grid_x - x_minus
    dy_minus = grid_y - y_minus
    
    # Distance from negative charge to each grid point
    r_minus = np.sqrt(dx_minus**2 + dy_minus**2)
    r_minus = np.maximum(r_minus, 1e-6)
    
    # Electric field components from negative charge
    # Note: Charge is -Q (negative)
    Ex_minus = k * (-Q) * dx_minus / (r_minus**3)
    Ey_minus = k * (-Q) * dy_minus / (r_minus**3)
    
    # 6. SUPERPOSITION: Total field is sum of both
    Ex_total = Ex_plus + Ex_minus
    Ey_total = Ey_plus + Ey_minus
    
    return Ex_total, Ey_total

def generate_dipole_dataset(
    n_circuits=100,
    grid_size=32,
    grid_range=0.3,
    l_bounds=(0.01, -0.1, -0.1, 1),
    u_bounds=(0.2,  0.1,  0.1, 100),
    field_clip=None,
    seed=42
):
    """
    Generate synthetic dipole dataset.

    Returns
    -------
    inputs  : (N_total, 6)
              [L, x_pos, y_pos, V, x, y]
    targets : (N_total, 2)
              [Ex, Ey]
    """

    np.random.seed(seed)

    # ======================================================
    # 1️⃣ Create spatial grid
    # ======================================================
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    x_flat = grid_x.flatten()
    y_flat = grid_y.flatten()

    n_points = grid_size * grid_size

    # ======================================================
    # 2️⃣ Sample dipole parameters (Latin Hypercube)
    # ======================================================
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    sample = sampler.random(n=n_circuits)
    sample = qmc.scale(sample, l_bounds, u_bounds)

    # ======================================================
    # 3️⃣ Compute fields
    # ======================================================
    results = []

    for L, x_pos, y_pos, V in sample:
        Ex, Ey = compute_dipole_field(L, x_pos, y_pos, V, grid_x, grid_y)
        results.append(np.stack([Ex, Ey], axis=-1))  # (grid, grid, 2)

    results_array = np.array(results)  # (n_circuits, grid, grid, 2)

    # Flatten fields
    results_array = results_array.reshape(-1, 2)  # (N_total, 2)

    # ======================================================
    # 4️⃣ Expand parameter inputs
    # ======================================================
    sample_extended = np.repeat(sample, repeats=n_points, axis=0)

    x_grid_extended = np.tile(x_flat, reps=n_circuits).reshape(-1, 1)
    y_grid_extended = np.tile(y_flat, reps=n_circuits).reshape(-1, 1)

    inputs = np.concatenate(
        [sample_extended, x_grid_extended, y_grid_extended],
        axis=1
    )

    targets = results_array

    # ======================================================
    # 5️⃣ Optional field clipping (for extreme outliers)
    # ======================================================
    if field_clip is not None:
        mask = (
            (np.abs(targets[:, 0]) <= field_clip) &
            (np.abs(targets[:, 1]) <= field_clip)
        )
        inputs = inputs[mask]
        targets = targets[mask]

    return inputs, targets

def inverse_signed_log_torch(y):
    return torch.sign(y) * torch.expm1(torch.abs(y))

def extract_circuit_batch(X, y, circuit_idx, n_points):
    start = circuit_idx * n_points
    end = start + n_points
    return X[start:end], y[start:end]

def predict_circuit(
    model,
    X_batch,
    y_scaler,
    device,
    apply_inverse_signed_log=True
):
    with torch.no_grad():

        print(f"The first ten inputs are: {X_batch[:10]}")
        preds = model(X_batch.to(device))

        if apply_inverse_signed_log:
            preds = inverse_signed_log_torch(preds)

        print(f"The first ten predictions are: {preds[:10]}")

        preds = preds.cpu().numpy()
        preds = y_scaler.inverse_transform(preds)
    
    return preds

def prepare_field_for_plot(
    predictions,
    targets,
    grid_size
):
    """
    Convert flat (N,2) arrays into (2, grid, grid)
    """

    preds = predictions.T.reshape(2, grid_size, grid_size)
    targs = targets.T.reshape(2, grid_size, grid_size)

    Ex_pred, Ey_pred = preds
    Ex_target, Ey_target = targs

    return Ex_pred, Ey_pred, Ex_target, Ey_target

def recover_physical_inputs(
    X_batch,
    X_scaler,
    grid_size
):
    X_not_scaled = X_scaler.inverse_transform(X_batch.cpu())

    L, x_q, y_q, V, x, y = X_not_scaled.T

    grid_x = x.reshape(grid_size, grid_size)
    grid_y = y.reshape(grid_size, grid_size)

    return L[0], x_q[0], y_q[0], V[0], grid_x, grid_y

def signed_log(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

# -----------------------------
# 1. Load model from MLflow
# -----------------------------

device = torch.device("cuda")
print(f"Using device: {device}")

model_name = "DipoleFieldModel_Spatial"
model_uri = f"models:/{model_name}/latest"
model = mlflow.pytorch.load_model(model_uri)

# class SurrogateModel(nn.Module):
#     def __init__(self,  hidden_dim=128, n_layers=3, input_dim = 6, output_dim = 2):

#         super().__init__()

#         layers_list = []
#         for i in range(n_layers):
#             layers_list.append(nn.Linear(hidden_dim, hidden_dim))
#             layers_list.append(nn.ReLU())
            
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             *[layer for layer in layers_list],
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.model(x)
    
# model = SurrogateModel().to(device=device)
# model.load_state_dict(torch.load('./models/dipole_surrogate_model.pth'))

model.eval().to(device)

print(model)

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

print(scaler_path)

X_scaler = joblib.load(f"{scaler_path}/x_scaler.save")
y_scaler = joblib.load(f"{scaler_path}/y_scaler.save")

# X_scaler = joblib.load('./models/x_scaler.save')
# y_scaler = joblib.load('./models/y_scaler.save')

print("Model and scalers loaded successfully!")


# Generate test dataset
grid_size = 32
n_circuits = 1
grid_range = 0.3

inputs, targets = generate_dipole_dataset(
    n_circuits=n_circuits,
    grid_size=grid_size,
    grid_range=grid_range,
    field_clip=3e4
)

print(inputs.shape)
print(targets.shape)

n_points = grid_size * grid_size
circuit_idx = 0

X_test = torch.tensor(X_scaler.transform(inputs), dtype=torch.float32)
y_test = torch.tensor(targets, dtype=torch.float32)

# Extract circuit
X_batch, y_batch = extract_circuit_batch(
    X_test, y_test, circuit_idx, n_points
)

# Prepare ground truth
y_true = y_batch.detach().cpu().numpy()  # safe copy, won't be affected by later transforms

# Prepare batch for model loss / MSE computation
y_scaled = signed_log(y_batch)
y_scaled = y_scaler.transform(y_scaled.detach().cpu().numpy())

# Predict
preds =  predict_circuit(
    model,
    X_batch,
    y_scaler,
    device
)

criterion = nn.MSELoss()
loss = criterion(torch.tensor(preds, dtype=torch.float32), y_batch)

print(f"The loss is: {loss.item():.4f}")

# Prepare fields
Ex_pred, Ey_pred, Ex_target, Ey_target = prepare_field_for_plot(
    preds,
    y_true,
    grid_size
)

# Recover geometry
L, x_q, y_q, V, grid_x, grid_y = recover_physical_inputs(
    X_batch,
    X_scaler,
    grid_size
)

# Plot predicted
fig, ax = plot_field_vectors(
    Ex_pred, Ey_pred,
    x_pos=x_q,
    y_pos=y_q,
    L=L,
    grid_x=grid_x,
    grid_y=grid_y,
    title="Predicted Electric Field"
)


# Plot target
fig1, ax1 = plot_field_vectors(
    Ex_target, Ey_target,
    x_pos=x_q,
    y_pos=y_q,
    L=L,
    grid_x=grid_x,
    grid_y=grid_y,
    title="Target Electric Field"
)

plt.show()

