import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from src.utils.paths import PROCESSED_DATA_DIR

from src.data.data import generate_input_features, compute_dipole_field


def get_train_val_loaders(
    batch_size=32,
    grid_size=32,
    grid_range=0.3,
    val_split=0.15,
    random_state=42,
):
    # -----------------------------
    # Create grid
    # -----------------------------
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    # -----------------------------
    # Generate input features
    # -----------------------------
    scaled_sample = generate_input_features()

    # -----------------------------
    # Generate dipole fields
    # -----------------------------
    results = []
    for params in scaled_sample:
        L, x_pos, y_pos, V = params
        Ex, Ey = compute_dipole_field(
            L, x_pos, y_pos, V, grid_x, grid_y
        )
        results.append((Ex, Ey))

    results_array = np.array(results)
    results_array = results_array.transpose(0, 2, 3, 1)

    input_tensor = torch.tensor(scaled_sample, dtype=torch.float32)
    output_tensor = torch.tensor(results_array, dtype=torch.float32).permute(0, 3, 1, 2)

    # -----------------------------
    # Train / validation split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        input_tensor,
        output_tensor,
        test_size=val_split,
        random_state=random_state,
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = torch.tensor(
        X_scaler.fit_transform(X_train.numpy()),
        dtype=torch.float32,
    )
    X_val_scaled = torch.tensor(
        X_scaler.transform(X_val.numpy()),
        dtype=torch.float32,
    )

    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_train_scaled = y_scaler.fit_transform(y_train_flat.numpy())
    y_train_scaled = torch.tensor(
        y_train_scaled.reshape(y_train.shape),
        dtype=torch.float32,
    )

    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    y_val_scaled = y_scaler.transform(y_val_flat.numpy())
    y_val_scaled = torch.tensor(
        y_val_scaled.reshape(y_val.shape),
        dtype=torch.float32,
    )

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        TensorDataset(X_train_scaled, y_train_scaled),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_scaled, y_val_scaled),
        batch_size=batch_size,
        shuffle=False,
    )

    import joblib
    joblib.dump(X_scaler, PROCESSED_DATA_DIR / "x_scaler.save")
    joblib.dump(y_scaler, PROCESSED_DATA_DIR / "y_scaler.save")

    print("Scalers saved correctly!")

    return train_loader, val_loader, X_scaler, y_scaler
