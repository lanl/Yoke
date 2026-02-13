"""This viewer makes temporal evolution plots for a single Cylex simulation.

Plot dimensions are to scale.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Utility Functions
# ==============================================================================


def load_sim_data(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Load .npz data and return a dictionary of variable arrays."""
    with np.load(npz_path) as dataset:
        return {key: np.asarray(dataset[key]) for key in dataset}


def extract_densities(
    dataset: dict[str, np.ndarray], keys: list[str]
) -> dict[str, float]:
    """Compute mean density values for a list of keys from dataset."""
    return {key: np.nanmean(dataset[key]) for key in keys}


def generate_contour_plot(
    dataset: dict[str, np.ndarray],
    extent: tuple[int, int, int, int],
    output_file: str | Path,
    variable_map: list[str],
) -> None:
    """Generate and save a contour plot for a set of variable arrays."""
    aspect_ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])
    z_arrays = [dataset[var] for var in variable_map]
    rows, cols = z_arrays[0].shape

    x_grid, y_grid = np.meshgrid(
        np.linspace(extent[0], extent[1], cols), np.linspace(extent[2], extent[3], rows)
    )

    fig, ax = plt.subplots(figsize=(10, 10 * aspect_ratio))
    for array, cmap in zip(z_arrays, ["Blues", "Purples", "YlOrBr", "Oranges"]):
        ax.contourf(x_grid, y_grid, array, levels=50, cmap=cmap)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Configuration
# ==============================================================================

SIM_ID = "00296"
DATA_DIR = Path(f"./cx241203_id{SIM_ID}/")
INDEX_RANGE = range(0, 11)

WALL = "density_wall"
BACKGROUND = "density_U.DU"
BOOSTER = "density_booster"
MAINCHARGE = "density_maincharge"
VARIABLE_KEYS = [WALL, BACKGROUND, BOOSTER, MAINCHARGE]

PLOT_EXTENT = (0, 400, 0, 1120)

file_paths = [
    DATA_DIR / f"cx241203_id{SIM_ID}_pvi_idx{idx:05d}.npz" for idx in INDEX_RANGE
]

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    for file_path in file_paths:
        try:
            sim_data = load_sim_data(file_path)

            df = pd.DataFrame(
                {
                    "Variable": list(sim_data.keys()),
                    "Size": [sim_data[key].shape for key in sim_data],
                }
            )
            print(f"\nData from {file_path}:\n{df}\n")

            densities = extract_densities(sim_data, VARIABLE_KEYS)
            for var, mean_val in densities.items():
                print(f"{var} mean density (g/cm³): {mean_val}")

            if all(key in sim_data for key in VARIABLE_KEYS):
                image_filename = file_path.stem + "_combined_contour.png"
                generate_contour_plot(
                    sim_data, PLOT_EXTENT, image_filename, VARIABLE_KEYS
                )
            else:
                print(f"One or more required variables not found in {file_path}.")

        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            print(f"Error processing file {file_path}: {error}")
