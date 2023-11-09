import numpy as np
import matplotlib.pyplot as plt


def plot_wafer_thickness_width_contour_map(
    wafer_diameter=100e-3,
    nominal_thickness=220e-9,
    min_thickness=210e-9,
    max_thickness=230e-9,
    nominal_width=500e-9,
    min_width=480e-9,
    max_width=520e-9,
    correlation_scale=0.1,
    seed=912,
    resolution=1e-3,
):
    
    """Plot a wafer thickness and width contour map with given parameters."""

    # Set seed for reproducibility
    np.random.seed(seed)

    # Calculate the grid size based on the wafer diameter and resolution
    wafer_size = int(wafer_diameter / resolution)

    def generate_correlated_noise(size, correlation_scale):
        """Generate a correlated random noise with given size and scale."""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)
        correlation_matrix = np.exp(-distance / correlation_scale)
        noise = np.random.randn(size, size)
        correlated_noise = np.fft.ifft2(np.fft.fft2(
            noise) * np.fft.fft2(correlation_matrix)).real
        return correlated_noise

    # Thickness variability
    correlated_noise_thickness = generate_correlated_noise(
        wafer_size, correlation_scale)
    normalized_noise_thickness = (correlated_noise_thickness - correlated_noise_thickness.min()) / (
        correlated_noise_thickness.max() - correlated_noise_thickness.min()
    )
    thickness_noise = min_thickness + \
        (max_thickness - min_thickness) * normalized_noise_thickness
    wafer_thickness = nominal_thickness + (thickness_noise - nominal_thickness)

    # Width variability
    correlated_noise_width = generate_correlated_noise(
        wafer_size, correlation_scale)
    normalized_noise_width = (correlated_noise_width - correlated_noise_width.min()) / (
        correlated_noise_width.max() - correlated_noise_width.min()
    )
    width_noise = min_width + (max_width - min_width) * normalized_noise_width
    wafer_width = nominal_width + (width_noise - nominal_width)

    # Generate coordinates for plotting
    coords = np.linspace(0, wafer_diameter, wafer_size)

    # Plot the wafer thickness contour map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.contourf(coords, coords, wafer_thickness, levels=np.linspace(
        min_thickness, max_thickness, 21), cmap='viridis')
    fig.colorbar(im1, ax=ax1, label='Thickness (m)')
    ax1.set_title('Wafer Thickness Contour Map')
    ax1.set_xlabel('X-axis (m)')
    ax1.set_ylabel('Y-axis (m)')
    ax1.axis('equal')

    im2 = ax2.contourf(coords, coords, wafer_width, levels=np.linspace(
        min_width, max_width, 21), cmap='viridis')
    fig.colorbar(im2, ax=ax2, label='Width (m)')
    ax2.set_title('Wafer Width Contour Map')
    ax2.set_xlabel('X-axis (m)')
    ax2.set_ylabel('Y-axis (m)')
    ax2.axis('equal')

    plt.show()


# Call the function with default parameters
plot_wafer_thickness_width_contour_map()
