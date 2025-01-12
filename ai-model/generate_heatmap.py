import os
import re

import cv2
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

band_paths = {
    "B4": "",
    "B5": "",
    "B10": "",
    "MLT": ""

}
def calculateHeatmap(folder_path):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith("MTL.txt"):
                band_paths["MLT"] = os.path.join(dirpath, filename)
            elif filename.endswith("B4.TIF"):
                band_paths["B4"] = os.path.join(dirpath, filename)
            elif filename.endswith("B5.TIF"):
                band_paths["B5"] = os.path.join(dirpath, filename)
            elif filename.endswith("B10.TIF"):
                band_paths["B10"] = os.path.join(dirpath, filename)

    with open(band_paths["MLT"], "r") as file:
        data = file.read()
        pattern = r'RADIANCE_MULT_BAND_10\s*=\s*(\S+)'
        match = re.search(pattern, data)
        ML = float(match.group(1))

        pattern = r'RADIANCE_ADD_BAND_10\s*=\s*(\S+)'
        match = re.search(pattern, data)
        AL = float(match.group(1))

        pattern = r'K1_CONSTANT_BAND_10\s*=\s*(\S+)'
        match = re.search(pattern, data)
        K1 = float(match.group(1))

        pattern = r'K2_CONSTANT_BAND_10\s*=\s*(\S+)'
        match = re.search(pattern, data)
        K2 = float(match.group(1))

    metadata = {
        'RADIANCE_MULT_BAND_10': ML,
        'RADIANCE_ADD_BAND_10': AL,
        'K1_CONSTANT_BAND_10': K1,
        'K2_CONSTANT_BAND_10': K2
    }

    return process_landsat_lst(
        band_paths["B4"],
        band_paths["B5"],
        band_paths["B10"],
        metadata,

    )

def calculate_ndvi(red_band, nir_band):
    """
    Calculate NDVI from red (Band 4) and NIR (Band 5) bands
    """
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10

    ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
    return ndvi


def calculate_emissivity(ndvi):
    """
    Calculate land surface emissivity using NDVI

    Based on the method from Sobrino et al. (2004):
    - NDVI < 0.2: bare soil
    - 0.2 ≤ NDVI ≤ 0.5: mixture of bare soil and vegetation
    - NDVI > 0.5: full vegetation
    """
    emissivity = np.zeros_like(ndvi)

    # Bare soil (NDVI < 0.2)
    bare_soil = ndvi < 0.2
    emissivity[bare_soil] = 0.973

    # Full vegetation (NDVI > 0.5)
    vegetation = ndvi > 0.5
    emissivity[vegetation] = 0.99

    # Mixed pixels (0.2 ≤ NDVI ≤ 0.5)
    mixed = np.logical_and(ndvi >= 0.2, ndvi <= 0.5)
    emissivity[mixed] = 0.973 + 0.017 * ((ndvi[mixed] - 0.2) / 0.3)

    return emissivity


def calculate_lst(thermal_band, emissivity, metadata):
    """
    Calculate Land Surface Temperature
    """
    # Get conversion parameters from metadata
    radiance_mult = float(metadata['RADIANCE_MULT_BAND_10'])
    radiance_add = float(metadata['RADIANCE_ADD_BAND_10'])
    k1 = float(metadata['K1_CONSTANT_BAND_10'])
    k2 = float(metadata['K2_CONSTANT_BAND_10'])

    # Convert DN to radiance
    radiance = (radiance_mult * thermal_band) + radiance_add

    # Convert to brightness temperature
    epsilon = 1e-10
    brightness_temp = k2 / np.log((k1 / (radiance + epsilon)) + 1)

    # Calculate LST using emissivity correction
    # LST = BT / (1 + (λ × BT / ρ) × ln(ε))
    # where λ = wavelength of emitted radiance (11.5 µm for Band 10)
    # ρ = h × c/σ (1.438 × 10^-2 m K)
    wavelength = 11.5 * 1e-6  # convert to meters
    rho = 1.438e-2

    lst = brightness_temp / (1 + (wavelength * brightness_temp / rho) * np.log(emissivity))

    return lst - 273.15  # Convert to Celsius


def process_landsat_lst(band4_path, band5_path, band10_path, metadata):
    # Process Landsat bands to calculate LST
    # Read all bands
    with rasterio.open(band4_path) as src:
        band4 = src.read(1)
        profile = src.profile

    with rasterio.open(band5_path) as src:
        band5 = src.read(1)

    with rasterio.open(band10_path) as src:
        band10 = src.read(1)

    # Create a mask for pure black pixels (where the pixel value is 0)
    black_pixels_mask = (band4 == 0) | (band5 == 0) | (band10 == 0)

    # Calculate NDVI, ignoring black pixels
    epsilon = 1e-10
    ndvi = np.zeros_like(band4, dtype=np.float32)  # Initialize with zeros
    ndvi[~black_pixels_mask] = (band5[~black_pixels_mask] - band4[~black_pixels_mask]) / \
                               (band5[~black_pixels_mask] + band4[~black_pixels_mask] + epsilon)

    # Calculate emissivity, ignoring black pixels
    emissivity = np.zeros_like(ndvi)
    bare_soil = (ndvi < 0.2) & ~black_pixels_mask
    vegetation = (ndvi > 0.5) & ~black_pixels_mask
    mixed = np.logical_and(ndvi >= 0.2, ndvi <= 0.5) & ~black_pixels_mask

    emissivity[bare_soil] = 0.973
    emissivity[vegetation] = 0.99
    emissivity[mixed] = 0.973 + 0.017 * ((ndvi[mixed] - 0.2) / 0.3)

    # Calculate LST, ignoring black pixels
    radiance_mult = float(metadata['RADIANCE_MULT_BAND_10'])
    radiance_add = float(metadata['RADIANCE_ADD_BAND_10'])
    k1 = float(metadata['K1_CONSTANT_BAND_10'])
    k2 = float(metadata['K2_CONSTANT_BAND_10'])

    # Convert DN to radiance
    radiance = (radiance_mult * band10) + radiance_add

    # Convert to brightness temperature
    brightness_temp = k2 / np.log((k1 / (radiance + epsilon)) + 1)

    # Calculate LST, ignoring black pixels
    wavelength = 11.5 * 1e-6
    rho = 1.438e-2
    lst = np.zeros_like(brightness_temp)
    lst[~black_pixels_mask] = brightness_temp[~black_pixels_mask] / \
                              (1 + (wavelength * brightness_temp[~black_pixels_mask] / rho) * np.log(
                                  emissivity[~black_pixels_mask]))

    lst_celsius = lst - 273.15

    # Get temperature range for scaling

    valid_lst = lst_celsius[~black_pixels_mask]  # Exclude black pixels
    temp_min = np.percentile(valid_lst, 1)
    temp_max = np.percentile(valid_lst, 99)
    print(f"Temperature range: {temp_min:.1f}°C to {temp_max:.1f}°C")

    # Create visualization with full resolution
    dpi = 100
    figure_size = (lst_celsius.shape[1] / dpi, lst_celsius.shape[0] / dpi)
    plt.figure(figsize=figure_size, dpi=dpi)

    # Custom colormap for temperature
    colors = ['black', 'darkblue', 'blue', 'royalblue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("temp_cmap", colors, N=n_bins)
    # Save colored visualization as .TIF
    profile.update(dtype=rasterio.uint8, count=3)
    # Normalize and apply colormap
    norm_temp = (lst_celsius - temp_min) / (temp_max - temp_min)
    colored = (cmap(norm_temp)[:, :, :3] * 255).astype(np.uint8)

    return colored, (temp_min, temp_max)


def main_heatmap(image_folder_path):

# for i in range(1, 21):
    return calculateHeatmap(image_folder_path)
