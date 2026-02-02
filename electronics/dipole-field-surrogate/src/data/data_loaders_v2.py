import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from src.utils.paths import PROCESSED_DATA_DIR

from src.data.data import generate_input_features, compute_dipole_field

def signed_log(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def get_train_val_loaders(
    batch_size=32,
    grid_size=32,
    grid_range=0.3,
    val_split=0.15,
    random_state=42,
    n_circuits=100,
    seed=42,
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
    sample = generate_input_features(num_samples=n_circuits, seed=seed)

    # -----------------------------
    # Generate dipole fields
    # -----------------------------
    results = []
    for params in sample:
        L, x_pos, y_pos, V = params
        Ex, Ey = compute_dipole_field(
            L, x_pos, y_pos, V, grid_x, grid_y
        )
        results.append((Ex, Ey))

    results_array = np.array(results)
    results_array = results_array.transpose(0, 2, 3, 1)
    results_array = results_array.reshape(-1, 2) # shape (n_samples, 2)

    x_grid = grid_x.flatten()
    y_grid = grid_y.flatten()
    x_grid_extended = np.tile(x_grid, reps=n_circuits).reshape(-1, 1)
    y_grid_extended = np.tile(y_grid, reps=n_circuits).reshape(-1, 1)

    sample_extended = np.repeat(sample, repeats=grid_size*grid_size, axis=0)

    input = np.concatenate([sample_extended, x_grid_extended, y_grid_extended], axis=1)
    target = results_array

    assert input.shape[0] == target.shape[0], f"Input and target first dimension must be equal"

    input_tensor = torch.tensor(input, dtype=torch.float32)
    output_tensor = torch.tensor(target, dtype=torch.float32)

    mask = (abs(output_tensor[:, 0]) <= 3e4) & (abs(output_tensor[:, 1]) <= 3e4)

    output_tensor = output_tensor[mask]
    input_tensor = input_tensor[mask]

    output_tensor = signed_log(output_tensor)

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
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = torch.tensor(
        X_scaler.fit_transform(X_train),
        dtype=torch.float32,
    )
    X_val_scaled = torch.tensor(
        X_scaler.transform(X_val),
        dtype=torch.float32,
    )

    y_train_scaled = torch.tensor(
        y_scaler.fit_transform(y_train),
        dtype=torch.float32,
    )


    y_val_scaled = torch.tensor(
        y_scaler.transform(y_val),
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

    print(f"First sample of train loader: {train_loader.dataset[0][0]}, {train_loader.dataset[0][1]}")
    print(f"First sample of val loader: {val_loader.dataset[0][0]}, {val_loader.dataset[0][1]}")

    import joblib
    joblib.dump(X_scaler, PROCESSED_DATA_DIR / "x_scaler.save")
    joblib.dump(y_scaler, PROCESSED_DATA_DIR / "y_scaler.save")

    print("Scalers saved correctly!")

    return train_loader, val_loader, X_scaler, y_scaler
