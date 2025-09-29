"""
YOLO Few-Shot Learning Pipeline

A comprehensive pipeline for few-shot object detection using YOLO models
with progressive curriculum learning and intelligent phase transitions.
"""

# Import main components for easy access
from .data import DatasetManager, create_rand_augment_transform
from .training import ProgressiveTrainer, ModelManager
from .evaluation import Evaluator
from .visualization import view_sample_images, save_training_plots, visualize_inference
from .utils import validate_base_config, load_and_validate_config, calculate_scaled_learning_rate

__version__ = "1.0.0"
__author__ = "YOLO Few-Shot Learning Team"

__all__ = [
    # Data components
    'DatasetManager',
    'create_rand_augment_transform',
    
    # Training components
    'ProgressiveTrainer',
    'ModelManager',
    
    # Evaluation components
    'Evaluator',
    
    # Visualization components
    'view_sample_images',
    'save_training_plots',
    'visualize_inference',
    
    # Utility functions
    'validate_base_config',
    'load_and_validate_config',
    'calculate_scaled_learning_rate',
]