"""
Configuration utilities for the YOLO few-shot pipeline.

Provides helpers to load, merge, and validate configuration dictionaries.
Designed to work with a global base config and stage-specific overrides.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_configs(base_config: Dict[str, Any], task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge a task-specific configuration into the base configuration.

    Returns a deep-merged dictionary where task values override base values.
    """
    merged = base_config.copy()
    
    def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> None:
        """Recursively merge override_dict into base_dict."""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_merge(merged, task_config)
    return merged


def load_and_merge_configs(base_config_path: str, task_config_path: str) -> Dict[str, Any]:
    """
    Load YAML files and return a merged configuration.
    """
    base_file = Path(base_config_path)
    task_file = Path(task_config_path)
    
    if not base_file.exists():
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")
    if not task_file.exists():
        raise FileNotFoundError(f"Task configuration file not found: {task_config_path}")
    
    with open(base_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    with open(task_file, 'r') as f:
        task_config = yaml.safe_load(f)
    
    merged_config = merge_configs(base_config, task_config)
    logger.info(f"✅ Configurations merged: {base_config_path} + {task_config_path}")
    return merged_config


def validate_base_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary for required keys and values.

    Note: This function validates the currently provided dictionary. When
    validating a base-only config, some stage-specific keys may be absent.
    """
    errors = []
    
    # Required top-level keys
    required_keys = [
        'model_name', 'num_classes', 'batch', 'epochs', 'lr0', 'lrf',
        'device', 'imgsz', 'seed'
    ]
    
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: '{key}'")
    
    # Validate specific values
    if 'model_name' in config:
        if not isinstance(config['model_name'], str) or not config['model_name']:
            errors.append("'model_name' must be a non-empty string")
    
    if 'num_classes' in config:
        if not isinstance(config['num_classes'], int) or config['num_classes'] <= 0:
            errors.append("'num_classes' must be a positive integer")
    
    if 'batch' in config:
        if not isinstance(config['batch'], int) or config['batch'] <= 0:
            errors.append("'batch' must be a positive integer")
    
    if 'epochs' in config:
        if not isinstance(config['epochs'], int) or config['epochs'] <= 0:
            errors.append("'epochs' must be a positive integer")
    
    if 'device' in config:
        device = config['device']
        if not isinstance(device, str):
            errors.append("'device' must be a string")
        elif device not in ['auto', 'cpu'] and not _is_valid_gpu_string(device):
            errors.append("'device' must be 'auto', 'cpu', or a valid GPU string like '0' or '0,1'")
    
    if 'imgsz' in config:
        if not isinstance(config['imgsz'], int) or config['imgsz'] <= 0:
            errors.append("'imgsz' must be a positive integer")
    
    if 'seed' in config:
        if not isinstance(config['seed'], int):
            errors.append("'seed' must be an integer")
    
    # Validate dataset configuration if present
    if 'dataset' in config:
        dataset_errors = _validate_dataset_config(config['dataset'])
        errors.extend(dataset_errors)
    
    # Validate randaugment configuration only if it's enabled
    if 'randaugment' in config and config['randaugment'].get('enabled', False):
        randaugment_errors = _validate_randaugment_config(config['randaugment'])
        errors.extend(randaugment_errors)
    
    return errors


def _is_valid_gpu_string(device_str: str) -> bool:
    """Validate whether a device string represents valid GPU IDs."""
    try:
        if ',' in device_str:
            # Multi-GPU format: "0,1" or "cuda:0,cuda:1"
            gpus = device_str.split(',')
            for gpu in gpus:
                gpu = gpu.strip()
                if gpu.startswith('cuda:'):
                    int(gpu.split(':')[1])
                else:
                    int(gpu)
        else:
            # Single GPU format: "0" or "cuda:0"
            if device_str.startswith('cuda:'):
                int(device_str.split(':')[1])
            else:
                int(device_str)
        return True
    except (ValueError, IndexError):
        return False


def _validate_dataset_config(dataset_config: Dict[str, Any]) -> List[str]:
    """Validate the main dataset configuration section (with `hf_repos`)."""
    errors = []
    # Only validate the main config; skip task-specific stubs.
    if 'hf_repos' not in dataset_config:
        return errors

    hf_repos = dataset_config['hf_repos']
    required_repos = ['coco_143', 'homeobjects_143', 'homeobjects_14']
    
    for repo in required_repos:
        if repo not in hf_repos:
            errors.append(f"Missing required repository: 'dataset.hf_repos.{repo}'")
        else:
            repo_config = hf_repos[repo]
            if 'repo_id' not in repo_config or 'filename' not in repo_config:
                errors.append(f"Repository '{repo}' missing 'repo_id' or 'filename'")
    return errors


def _validate_randaugment_config(randaugment_config: Dict[str, Any]) -> List[str]:
    """Validate the randaugment configuration section."""
    errors = []
    
    # Check if enabled field exists and is boolean
    if 'enabled' not in randaugment_config:
        errors.append("Missing randaugment configuration: 'enabled'")
    elif not isinstance(randaugment_config['enabled'], bool):
        errors.append("'randaugment.enabled' must be a boolean")
    
    # Only validate other fields if RandAugment is enabled
    if randaugment_config.get('enabled', False):
        required_keys = ['num_ops', 'magnitude', 'mstd']
        for key in required_keys:
            if key not in randaugment_config:
                errors.append(f"Missing randaugment configuration: '{key}'")
        
        if 'num_ops' in randaugment_config:
            if not isinstance(randaugment_config['num_ops'], int) or randaugment_config['num_ops'] < 0:
                errors.append("'randaugment.num_ops' must be a non-negative integer")
        
        if 'magnitude' in randaugment_config:
            if not isinstance(randaugment_config['magnitude'], int) or randaugment_config['magnitude'] < 0:
                errors.append("'randaugment.magnitude' must be a non-negative integer")
        
        if 'mstd' in randaugment_config:
            if not isinstance(randaugment_config['mstd'], (int, float)) or randaugment_config['mstd'] < 0:
                errors.append("'randaugment.mstd' must be a non-negative number")
    
    return errors


def validate_config_file(config_path: Path) -> bool:
    """
    Validate a configuration file on disk by loading and checking it.
    """
    try:
        import yaml
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        errors = validate_base_config(config)
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info(f"✅ Configuration file validated successfully: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration file {config_path}: {e}")
        return False


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a configuration file, returning the parsed dictionary.
    """
    import yaml
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    errors = validate_base_config(config)
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    logger.info(f"✅ Configuration loaded and validated: {config_path}")
    return config


def calculate_scaled_learning_rate(base_lr: float, base_batch_size: int, new_batch_size: int) -> float:
    """
    Calculate a scaled learning rate using the Linear Scaling Rule.
    """
    if base_batch_size <= 0 or new_batch_size <= 0:
        raise ValueError("Batch sizes must be positive integers")
    
    scaling_factor = new_batch_size / base_batch_size
    return base_lr * scaling_factor


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a concise summary of the configuration for verification.
    """
    logger.info("📋 Configuration Summary:")
    logger.info(f"  Model: {config.get('model_name', 'N/A')}")
    logger.info(f"  Classes: {config.get('num_classes', 'N/A')}")
    logger.info(f"  Device: {config.get('device', 'N/A')}")
    logger.info(f"  Batch Size: {config.get('batch', 'N/A')}")
    logger.info(f"  Epochs: {config.get('epochs', 'N/A')}")
    logger.info(f"  Image Size: {config.get('imgsz', 'N/A')}")
    logger.info(f"  Seed: {config.get('seed', 'N/A')}")
    
    # Show learning rate scaling information
    batch_size = config.get('batch', 0)
    lr0 = config.get('lr0', 0)
    if batch_size > 0 and lr0 > 0:
        # Assume base batch size of 64 for single GPU
        base_batch_size = 64
        if batch_size != base_batch_size:
            base_lr = lr0 * base_batch_size / batch_size
            logger.info(f"  Learning Rate: {lr0} (scaled from {base_lr:.4f} for batch size {batch_size})")
        else:
            logger.info(f"  Learning Rate: {lr0}")
    
    if 'randaugment' in config:
        ra_config = config['randaugment']
        logger.info(f"  RandAugment: {'Enabled' if ra_config.get('enabled', False) else 'Disabled'}")
        if ra_config.get('enabled', False):
            logger.info(f"    Operations: {ra_config.get('num_ops', 'N/A')}")
            logger.info(f"    Magnitude: {ra_config.get('magnitude', 'N/A')}")
