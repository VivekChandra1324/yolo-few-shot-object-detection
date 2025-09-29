# src/training/model_manager.py

import logging
import torch
from typing import Dict, Any
from ultralytics import YOLO

# Project-specific modules
from src.data.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model lifecycle helper for Ultralytics YOLO.

    Responsibilities:
    - Load a specified checkpoint or model name.
    - Resolve and pin the primary compute device.
    - Freeze an initial slice of the backbone for staged fine-tuning.
    - Provide lightweight model introspection for logging.
    """

    def __init__(self, config: Dict[str, Any], dataset_manager: DatasetManager):
        self.config = config
        self.dm = dataset_manager
        self.device = self._resolve_device(self.config.get('device', 'auto'))
        self.model: Any = None
        logger.info(f"ModelManager initialized on device '{self.device}'.")

    def _resolve_device(self, device_str: str) -> torch.device:
        """Convert config device string (e.g., '0,1' or 'cpu') into torch.device."""
        if device_str in ('auto', '', None):
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'cpu' in device_str:
            return torch.device('cpu')

        # Take the first GPU from a multi-GPU string like '0,1'
        primary_device_id = device_str.split(',')[0].strip()

        # Construct the full 'cuda:X' string if a digit is provided
        if primary_device_id.isdigit():
            return torch.device(f'cuda:{primary_device_id}')

        # Fallback for strings that might already be in 'cuda:0' format
        return torch.device(primary_device_id)

    def create_model(self) -> YOLO:
        """Load YOLO, then freeze the requested number of backbone layers."""
        try:
            model_name = self.config['model_name']
            logger.info(f"Loading pre-trained model: {model_name}...")
            
            self.model = YOLO(model_name)
            num_layers_to_freeze = self.config.get('freeze', 10)
            if num_layers_to_freeze > 0:
                self._freeze_layers(num_layers_to_freeze)
            
            logger.info(f"✅ Model '{model_name}' is configured and ready for training.")
            self.log_model_info()
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Failed to create and configure model: {e}", exc_info=True)
            raise

    def _freeze_layers(self, num_layers_to_freeze: int):
        """Disable gradients for the first N modules of the backbone in-place."""
        logger.info(f"Freezing the first {num_layers_to_freeze} layers of the model backbone...")
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            backbone = self.model.model.model
            num_layers_to_freeze = min(num_layers_to_freeze, len(backbone))

            for i, module in enumerate(backbone):
                if i < num_layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True

            logger.info(f"Successfully froze {num_layers_to_freeze} layers.")
        else:
            logger.warning("Could not access model backbone to freeze layers.")

    def log_model_info(self):
        """Log total, trainable, and frozen parameter counts for quick inspection."""
        if self.model is None:
            logger.warning("Model not loaded. Cannot log info.")
            return
            
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            logger.info("--- Model Parameters ---")
            logger.info(f"  - Total:     {total_params:,}")
            logger.info(f"  - Trainable: {trainable_params:,}")
            logger.info(f"  - Frozen:    {frozen_params:,}")
            logger.info("------------------------")
        except Exception as e:
            logger.warning(f"Could not calculate model parameters: {e}")