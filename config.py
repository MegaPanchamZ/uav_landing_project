# config.py
"""
Configuration file for UAV Landing Zone Detection System
All constants and tuning parameters are centralized here for easy adjustment.
"""

# --- Neural Network & Camera ---
MODEL_PATH = "bisenetv2_uav.onnx"  # Start with ONNX, can upgrade to .engine later
INPUT_RESOLUTION = (512, 512)  # W, H - model input size
CAMERA_RESOLUTION = (1920, 1080)  # W, H - actual camera resolution

# --- Class Definitions (MUST match your training) ---
CLASS_MAP = {
    0: "background",
    1: "safe_flat_surface",
    2: "unsafe_uneven_surface", 
    3: "low_obstacle",
    4: "high_obstacle"
}

# Reverse mapping for convenience
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}

# The class ID that represents safe landing zones
SAFE_LANDING_CLASS_ID = CLASS_NAME_TO_ID["safe_flat_surface"]
HIGH_OBSTACLE_CLASS_ID = CLASS_NAME_TO_ID["high_obstacle"]
LOW_OBSTACLE_CLASS_ID = CLASS_NAME_TO_ID["low_obstacle"]

# --- Symbolic Engine Rules ---
# Minimum area in pixels for a zone to be considered viable
MIN_LANDING_AREA_PIXELS = 3000

# Minimum clearance in pixels from obstacles
HIGH_OBSTACLE_CLEARANCE = 30  # pixels clearance from high obstacles
LOW_OBSTACLE_CLEARANCE = 15   # pixels clearance from low obstacles

# Shape analysis parameters
MIN_ASPECT_RATIO = 0.3  # Reject very elongated zones
MAX_ASPECT_RATIO = 3.0  # Reject very elongated zones
MIN_SOLIDITY = 0.6      # Reject zones with too many holes/concavities

# --- Temporal Stability Parameters ---
# Zone must be valid for this many frames in the last N frames to be "confirmed"
TEMPORAL_WINDOW_SIZE = 15
TEMPORAL_CONFIRMATION_THRESHOLD = 10  # Must be valid in at least this many recent frames

# --- Candidate Scoring Weights ---
W_AREA = 0.4        # Prioritize larger areas
W_CENTER = 0.3      # Prioritize areas near the center of the screen
W_SHAPE = 0.2       # Prioritize well-shaped zones (circular/square)
W_CLEARANCE = 0.1   # Prioritize zones with more clearance

# --- Image Processing Parameters ---
# Gaussian blur kernel size for noise reduction
BLUR_KERNEL_SIZE = 5

# Morphological operations for cleaning up segmentation
MORPH_KERNEL_SIZE = 3
MORPH_ITERATIONS = 2

# --- Visualization Parameters ---
ZONE_COLOR = (0, 255, 0)      # Green for valid zones
OBSTACLE_COLOR = (0, 0, 255)  # Red for obstacles
CENTER_COLOR = (255, 255, 255) # White for zone centers
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# --- Debug and Logging ---
DEBUG_MODE = True
SAVE_DEBUG_FRAMES = False
DEBUG_OUTPUT_DIR = "debug_output"

# --- Performance Monitoring ---
TARGET_FPS = 30
PERFORMANCE_WINDOW = 100  # Frames to average for FPS calculation

# --- GPS-Free Navigation Parameters ---
# Visual Odometry
ENABLE_VISUAL_ODOMETRY = True
FEATURE_DETECTOR_TYPE = "ORB"  # ORB, SIFT, SURF
MAX_FEATURES = 1000
MATCH_DISTANCE_THRESHOLD = 50

# Camera Calibration (These should be calibrated for your specific camera)
# Default values for a typical camera - MUST be calibrated for production use
CAMERA_MATRIX = [
    [800.0, 0.0, 320.0],    # fx, 0, cx
    [0.0, 800.0, 240.0],    # 0, fy, cy
    [0.0, 0.0, 1.0]         # 0, 0, 1
]
DISTORTION_COEFFICIENTS = [0.0, 0.0, 0.0, 0.0, 0.0]  # [k1, k2, p1, p2, k3]

# Altitude Estimation
INITIAL_ALTITUDE_ESTIMATE = 10.0  # meters
MIN_ALTITUDE_ESTIMATE = 2.0       # meters
MAX_ALTITUDE_ESTIMATE = 50.0      # meters
ALTITUDE_SMOOTHING_FACTOR = 0.1   # Lower = more smoothing

# Motion Estimation
MOTION_HISTORY_SIZE = 10
MIN_MOTION_CONFIDENCE = 0.3
RANSAC_THRESHOLD = 1.0
RANSAC_CONFIDENCE = 0.999

# Relative Positioning
POSITION_CONTROL_GAIN = 0.5
MAX_APPROACH_SPEED = 2.0    # m/s
MAX_PRECISION_SPEED = 0.5   # m/s

# Landing Phases
APPROACH_ALTITUDE = 5.0     # meters - switch to approach mode
PRECISION_ALTITUDE = 2.0    # meters - switch to precision mode
LANDING_THRESHOLD = 0.5     # meters - final landing threshold

# Emergency safety parameters
MIN_SAFE_ALTITUDE = 0.5         # Minimum altitude before emergency stop (meters)
MAX_FLIGHT_ALTITUDE = 50.0      # Maximum allowed flight altitude (meters)  
GEOFENCE_RADIUS = 100.0         # Circular safety boundary (meters)
MAX_TARGET_LOSS_FRAMES = 10     # Emergency hover after losing target
EMERGENCY_HOVER_TIMEOUT = 30.0  # Emergency hover duration (seconds)

# =====================================================
# FINE-TUNING CONFIGURATION
# =====================================================

# Model fine-tuning pipeline settings
FINE_TUNING_CONFIG = {
    # Step 1: Base training on DroneDeploy  
    "step1": {
        "dataset": "dronedeploy",
        "epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 8,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "augmentations": ["flip", "rotate", "color_jitter"]
    },
    
    # Step 2: Intermediate fine-tuning on UDD-6
    "step2": {
        "dataset": "udd6", 
        "epochs": 30,
        "learning_rate": 5e-4,
        "batch_size": 12,
        "weight_decay": 1e-4,
        "scheduler": "step",
        "freeze_layers": ["detail_branch.0", "detail_branch.1"]  # Freeze early layers
    },
    
    # Step 3: Task-specific fine-tuning
    "step3": {
        "dataset": "udd6",
        "epochs": 20, 
        "learning_rate": 1e-4,
        "batch_size": 16,
        "weight_decay": 5e-5,
        "scheduler": "cosine_warm_restarts",
        "class_weights": [1.0, 2.0, 1.5, 2.0, 3.0, 1.0]  # Emphasize landing-relevant classes
    }
}

# Target class system for UAV landing
TARGET_CLASSES = {
    0: "background",    # Non-relevant background areas
    1: "suitable",      # Flat, clear areas ideal for landing  
    2: "marginal",      # Areas that might work (low vegetation, rough ground)
    3: "obstacles",     # Buildings, structures, tall objects to avoid
    4: "unsafe",        # Water, vehicles, moving objects, steep terrain  
    5: "unknown"        # Uncertain/unclassified areas requiring caution
}

# Class mapping from original datasets to our target system
DATASET_CLASS_MAPPINGS = {
    "dronedeploy": {
        0: 0,  # background -> background
        1: 3,  # building -> obstacles  
        2: 3,  # clutter -> obstacles
        3: 2,  # vegetation -> marginal (if relatively flat)
        4: 4,  # water -> unsafe
        5: 1,  # ground -> suitable
        6: 4   # car -> unsafe
    },
    
    "udd6": {
        0: 0,  # other -> background
        1: 3,  # facade -> obstacles
        2: 1,  # road -> suitable
        3: 2,  # vegetation -> marginal
        4: 4,  # vehicle -> unsafe  
        5: 3   # roof -> obstacles
    }
}

# Model architecture settings
MODEL_CONFIG = {
    "input_size": (512, 512),       # Standard input resolution
    "num_classes": 6,               # Our 6-class landing system
    "backbone": "bisenetv2",        # BiSeNetV2 for real-time performance
    "pretrained_weights": "cityscapes",  # Base pre-trained weights
    "output_stride": 4,             # Output downsampling factor
    "aux_loss_weight": 0.4,         # Weight for auxiliary losses during training
}

# Data augmentation settings for aerial imagery
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,         # 50% chance
    "vertical_flip": 0.3,           # 30% chance (less common for aerial)
    "rotation_degrees": 360,        # Full rotation for aerial views
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3, 
        "saturation": 0.3,
        "hue": 0.1
    },
    "gaussian_blur": {
        "kernel_size": (3, 7),
        "sigma": (0.1, 2.0),
        "probability": 0.2
    }
}

# Landing phase control parameters (refined)
LANDING_PHASES = {
    "SEARCH": {
        "altitude_range": (5.0, 50.0),     # Search phase altitude
        "max_velocity": 2.0,               # m/s
        "descent_rate": 0.3,               # m/s
        "position_gain": 1.2,              # Aggressive positioning
        "required_confidence": 0.6         # Minimum confidence to proceed
    },
    
    "APPROACH": {
        "altitude_range": (2.0, 5.0),      # Approach phase altitude  
        "max_velocity": 1.5,               # m/s
        "descent_rate": 0.2,               # m/s
        "position_gain": 1.0,              # Standard positioning
        "required_confidence": 0.7         # Higher confidence required
    },
    
    "PRECISION": {
        "altitude_range": (0.5, 2.0),      # Precision phase altitude
        "max_velocity": 0.5,               # m/s
        "descent_rate": 0.1,               # m/s  
        "position_gain": 0.6,              # Gentle positioning
        "required_confidence": 0.8         # High confidence required
    },
    
    "LANDING": {
        "altitude_range": (0.0, 0.5),      # Final landing phase
        "max_velocity": 0.2,               # m/s
        "descent_rate": 0.05,              # m/s
        "position_gain": 0.3,              # Very gentle
        "required_confidence": 0.9         # Maximum confidence required
    }
}

# ROS integration parameters
ROS_CONFIG = {
    "image_topic": "/camera/image_raw",
    "altitude_topic": "/mavros/global_position/rel_alt", 
    "velocity_command_topic": "/mavros/setpoint_velocity/cmd_vel",
    "landing_status_topic": "/uav_landing/status",
    "detection_result_topic": "/uav_landing/detection",
    "visualization_topic": "/uav_landing/visualization",
    "queue_size": 10,
    "processing_rate": 30.0             # Hz - target processing frequency
}

# Flight Controller
MAX_COMMAND_DURATION = 10.0  # seconds
SIMULATION_UPDATE_RATE = 10  # Hz
MAX_VELOCITY_LIMIT = 5.0     # m/s
MAX_ACCELERATION = 2.0       # m/s^2
