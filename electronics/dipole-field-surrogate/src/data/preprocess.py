# %% 
from utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the generated data
circuit_data = np.load(RAW_DATA_DIR / "circuit_data.npz")

inputs = circuit_data["inputs"]
outputs = circuit_data["outputs"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
)

# Normalize the input features
x_scaler = MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)

y_scaler = MinMaxScaler()
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
y_train_scaled = y_scaler.fit_transform(y_train_reshaped)
y_train_scaled = y_train_scaled.reshape(y_train.shape)

y_val_reshaped = y_val.reshape(y_val.shape[0], -1)
y_val_scaled = y_scaler.transform(y_val_reshaped)
y_val_scaled = y_val_scaled.reshape(y_val.shape)

# Save the preprocessed data
np.savez(
    PROCESSED_DATA_DIR / "circuit_data_preprocessed.npz",
    X_train=X_train_scaled,
    X_val=X_val_scaled,
    y_train=y_train_scaled,
    y_val=y_val_scaled,
    x_scaler_min=x_scaler.data_min_,
    x_scaler_max=x_scaler.data_max_,
    y_scaler_min=y_scaler.data_min_,
    y_scaler_max=y_scaler.data_max_,
)

print(f"Preprocessed data saved to {PROCESSED_DATA_DIR / 'circuit_data_preprocessed.npz'}!")

# Save the scalers separately
import joblib
joblib.dump(x_scaler, PROCESSED_DATA_DIR / "x_scaler.save")
joblib.dump(y_scaler, PROCESSED_DATA_DIR / "y_scaler.save")
