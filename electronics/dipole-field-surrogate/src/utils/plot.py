import matplotlib.pyplot as plt
import numpy as np

# Create grid
grid_size = 32
grid_range = 0.3
x = np.linspace(-grid_range, grid_range, grid_size)
y = np.linspace(-grid_range, grid_range, grid_size)
grid_x, grid_y = np.meshgrid(x, y)


def plot_field_vectors(Ex, Ey, L, x_pos, y_pos, grid_x=grid_x, grid_y=grid_y,
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
