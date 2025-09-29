# src/data/augmentation.py

"""
Data augmentation utilities based on the `timm` library's RandAugment.

Overview
--------
- Exposes a factory function that builds a RandAugment transform object.
- RandAugment is the only augmentation used for novel classes in this project.
- The transform is applied on-the-fly by the ProgressiveTrainer callback so
  the same image can receive different augmentations across epochs.
"""

import logging
from typing import Callable, Optional, Union, Dict
from timm.data.auto_augment import rand_augment_transform

# --- Setup Logging ---
logger = logging.getLogger(__name__)


def create_rand_augment_transform(
    config: Optional[Dict] = None,
    *,
    magnitude: Optional[int] = None,
    num_ops: Optional[int] = None,
    mstd: Optional[float] = None,
) -> Callable:
    """
    Create a RandAugment transform object using `timm`.

    This reads the parameters from a provided configuration dictionary and
    constructs the appropriate transform string for `timm`. RandAugment applies
    random combinations of augmentation operations at runtime.

    Args:
        config (Optional[Dict]): A dictionary containing the 'randaugment' settings.
                                Expected keys are 'num_ops', 'magnitude',
                                and 'mstd' (magnitude standard deviation).
        magnitude (Optional[int]): Magnitude/intensity of augmentations.
                                  Overrides config if provided.
        num_ops (Optional[int]): Number of augmentation operations to apply.
                                Overrides config if provided.
        mstd (Optional[float]): Standard deviation for magnitude randomization.
                               Overrides config if provided.

    Returns:
        Callable: A callable that accepts a PIL.Image input and returns an
                  augmented PIL.Image.

    Raises:
        ValueError: If required parameters are missing from the configuration.
        Exception: If the timm library fails to create the transform.
    """
    rand_config = {}
    if isinstance(config, dict):
        # Accept either nested under 'randaugment' or flat dict
        if 'randaugment' in config and isinstance(config['randaugment'], dict):
            rand_config = config['randaugment']
        else:
            rand_config = config

    # Prefer explicit keyword args, falling back to the config
    resolved_num_ops = num_ops if num_ops is not None else rand_config.get('num_ops')
    resolved_magnitude = magnitude if magnitude is not None else rand_config.get('magnitude')
    magnitude_std = mstd if mstd is not None else rand_config.get('mstd')

    if resolved_num_ops is None or resolved_magnitude is None or magnitude_std is None:
        raise ValueError("RandAugment config must define 'num_ops', 'magnitude', and 'mstd'")

    # Construct the configuration string required by timm.
    # Format: 'rand-m{magnitude}-n{num_ops}-mstd{std_dev}'
    config_str = f'rand-m{resolved_magnitude}-n{resolved_num_ops}-mstd{magnitude_std}'

    # The 'hparams' dictionary holds additional parameters.
    # 'img_mean' provides the fill color for geometric ops; gray is standard.
    hparams = {'img_mean': (128, 128, 128)}
    
    try:
        timm_transform = rand_augment_transform(config_str, hparams)
        logger.info(f"✅ Successfully created timm RandAugment transform with config: '{config_str}'")
        return timm_transform
    except Exception as e:
        logger.error(f"❌ Failed to create timm RandAugment transform: {e}")
        raise