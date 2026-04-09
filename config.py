"""
PV Layout Design System - Configuration Module
==============================================
Central configuration for Malaysian rooftop solar PV layout optimization.
All constants, panel specs, irradiance data, and system parameters.
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OBSTACLE_DATASET_DIR = os.path.join(DATASET_DIR, "obstacles")
ROOF_DATASET_DIR = os.path.join(DATASET_DIR, "roofs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ============================================================
# SOLAR PANEL SPECIFICATIONS (620 Wp Module)
# ============================================================
PANEL_SPEC = {
    "rated_power_wp": 620,           # Watt-peak
    "length_m": 2.384,               # meters (longer side)
    "width_m": 1.303,                # meters (shorter side)
    "efficiency": 0.213,             # ~21.3% module efficiency
    "temp_coefficient": -0.0035,     # %/°C power temperature coefficient
    "noct": 45,                      # Nominal Operating Cell Temperature (°C)
    "voltage_mpp": 41.65,            # Vmp at STC (V)
    "current_mpp": 14.89,            # Imp at STC (A)
    "voltage_oc": 49.50,             # Voc at STC (V)
    "current_sc": 15.78,             # Isc at STC (A)
}

# ============================================================
# LAYOUT CONSTRAINTS
# ============================================================
LAYOUT = {
    "min_edge_setback_m": 0.5,       # Min distance from roof edge (m)
    "inter_row_gap_m": 0.3,          # Gap between panel rows (m)
    "inter_col_gap_m": 0.02,         # Gap between panel columns (m)
    "obstacle_buffer_m": 0.5,        # Clearance around obstacles (m)
    "walkway_width_m": 0.6,          # Maintenance walkway width (m)
    "tilt_angle_deg": 10,            # Default tilt for Malaysia (degrees)
    "azimuth_deg": 180,              # Default south-facing (degrees from N)
    "max_string_length": 20,         # Max panels per string
    "min_string_length": 8,          # Min panels per string
}

# ============================================================
# MALAYSIAN IRRADIANCE DATA (kWh/m²/day by region)
# Source: Malaysian Meteorological Department & SEDA Malaysia
# ============================================================
MALAYSIA_IRRADIANCE = {
    # Region: (avg_daily_ghi, peak_sun_hours, avg_temp_C)
    "Peninsular_West":   (4.8, 4.5, 28.5),
    "Peninsular_East":   (5.0, 4.7, 27.8),
    "Kuala_Lumpur":      (4.6, 4.3, 28.0),
    "Penang":            (5.1, 4.8, 28.2),
    "Johor":             (4.7, 4.4, 27.5),
    "Sabah":             (5.2, 4.9, 27.0),
    "Sarawak":           (4.9, 4.6, 27.2),
    "Perak":             (5.0, 4.7, 28.0),
    "Kedah":             (5.3, 5.0, 28.5),
    "Melaka":            (4.8, 4.5, 28.3),
    "Default":           (4.8, 4.5, 28.0),
}

# ============================================================
# PERFORMANCE ASSUMPTIONS
# ============================================================
PERFORMANCE = {
    "system_losses": 0.14,           # 14% total system losses
    "inverter_efficiency": 0.97,     # 97% inverter efficiency
    "soiling_loss": 0.03,            # 3% soiling/dust
    "mismatch_loss": 0.02,           # 2% module mismatch
    "wiring_loss": 0.02,             # 2% DC/AC wiring
    "degradation_year1": 0.02,       # 2% first-year degradation
    "degradation_annual": 0.005,     # 0.5% annual degradation after year 1
    "performance_ratio": 0.80,       # Overall performance ratio
    "availability": 0.98,            # 98% system availability
}

# ============================================================
# ELECTRICITY TARIFF (ATAP / NEM rates in Malaysia)
# ============================================================
TARIFF = {
    "nem_rate_myr_kwh": 0.312,       # NEM displacement rate (MYR/kWh)
    "feed_in_tariff_myr_kwh": 0.218, # Feed-in tariff if applicable
    "system_cost_myr_per_wp": 3.50,  # Avg installed cost (MYR/Wp)
    "annual_maintenance_myr": 500,    # Annual O&M cost estimate
    "system_lifetime_years": 25,      # Expected system lifetime
}

# ============================================================
# IMAGE PROCESSING PARAMETERS
# ============================================================
IMAGE_PROCESSING = {
    "input_size": (640, 640),         # Standard processing size
    "cnn_input_size": (128, 128),     # CNN classifier input size
    "roof_detection_blur": 5,         # Gaussian blur kernel
    "canny_low": 50,                  # Canny edge low threshold
    "canny_high": 150,                # Canny edge high threshold
    "min_contour_area_ratio": 0.05,   # Min contour area as ratio of image
    "obstacle_min_area_ratio": 0.005, # Min obstacle area ratio
    "obstacle_max_area_ratio": 0.15,  # Max obstacle area ratio
}

# ============================================================
# CNN MODEL PARAMETERS
# ============================================================
CNN_CONFIG = {
    "num_classes": 5,                 # water_tank, parapet, vent, ac_unit, none
    "class_names": ["water_tank", "parapet", "vent", "ac_unit", "none"],
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.001,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "augmentation": True,
    "early_stopping_patience": 10,
    "model_save_path": os.path.join(MODEL_DIR, "obstacle_classifier.pth"),
}
