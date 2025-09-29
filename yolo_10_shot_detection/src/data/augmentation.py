# src/data/augmentation.py

"""
RandAugment utilities (backed by the `timm` library).

This module exposes a single factory that constructs a RandAugment transform
configured from project settings. The transform is applied on-the-fly to
novel-class images by the training callback, providing stochastic, in-memory
augmentation per epoch without altering datasets on disk.
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
    Create a `timm`-based RandAugment transform.

    - Reads settings from `config['randaugment']` when present or from the
      provided keyword overrides.
    - Expects PIL Images as input and returns augmented PIL Images.

    Args:
        config: Project configuration dict or a dict containing RandAugment keys.
        magnitude: Augmentation magnitude; overrides config when provided.
        num_ops: Number of random ops per image; overrides config when provided.
        mstd: Magnitude standard deviation; overrides config when provided.

    Returns:
        A callable transform to apply RandAugment to PIL Images.

    Raises:
        ValueError: Missing one of required keys: 'num_ops', 'magnitude', 'mstd'.
        Exception: Underlying `timm` failure while constructing the transform.
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

    # The 'hparams' dictionary can be used to pass additional parameters.
    # 'img_mean' is used for the fill color of geometric transforms.
    # (128, 128, 128) is a standard gray value.
    hparams = {'img_mean': (128, 128, 128)}
    
    try:
        timm_transform = rand_augment_transform(config_str, hparams)
        logger.info(f"✅ Successfully created timm RandAugment transform with config: '{config_str}'")
        return timm_transform
    except Exception as e:
        logger.error(f"❌ Failed to create timm RandAugment transform: {e}")
        raise