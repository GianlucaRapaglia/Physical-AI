import numpy as np
from scipy.stats import qmc
from src.utils.paths import RAW_DATA_DIR

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

def generate_input_features(num_samples=10000, dimensions=4, seed=42):
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    sample = sampler.random(n=num_samples)

    l_bounds = [0.01, -0.1, -0.1, 1.0]  # Lower bounds for L, x_pos, y_pos, V
    u_bounds = [0.2, 0.1, 0.1, 100.0]  # Upper bounds for L, x_pos, y_pos, V

    sample = qmc.scale(sample, l_bounds, u_bounds)

    return sample


def main():
    # Create grid
    grid_size = 32
    grid_range = 0.3
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)

    # Generate input features
    scaled_sample = generate_input_features()

    # Generate fields to be predicted
    results = []
    for params in scaled_sample:
        L, x_pos, y_pos, V = params
        Ex, Ey = compute_dipole_field(L, x_pos, y_pos, V, grid_x, grid_y)
        results.append((Ex, Ey))

    input_array = np.stack(scaled_sample, axis=0)
    output_array = np.stack(
        [np.stack((Ex, Ey), axis=0) for Ex, Ey in results], axis=0
    )

    # Save data
    np.savez(
        RAW_DATA_DIR / "circuit_data.npz",
        inputs=input_array,
        outputs=output_array,
        input_names=["L", "x_pos", "y_pos", "V"],
        output_names=["Ex", "Ey"]
    )

    print(f"Data saved to {RAW_DATA_DIR / 'circuit_data.npz'}!")

if __name__ == "__main__":
    main()
