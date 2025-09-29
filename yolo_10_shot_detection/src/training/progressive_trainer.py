# src/training/progressive_trainer.py

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

# Import all necessary modules from our project
from src.data.augmentation import create_rand_augment_transform
from src.data.dataset_manager import DatasetManager
from src.evaluation.evaluator import Evaluator
from src.training.model_manager import ModelManager
from src.visualization.plotter import save_training_plots

logger = logging.getLogger(__name__)


class ProgressiveCallbackTrainer(DetectionTrainer):
    """
    Ultralytics trainer extension with progressive training callbacks.

    Injects custom behavior into the standard training loop:
    - Apply RandAugment to novel-only images at batch-start, in-memory.
    - Trigger periodic validation at a configured interval.
    - Early-terminate a phase when mAP@0.5 passes a threshold.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_progressive_trainer: Optional['ProgressiveTrainer'] = None

    def on_train_batch_start(self, batch):
        """Apply RandAugment to images labeled novel for this phase."""
        if not self.custom_progressive_trainer or not self.custom_progressive_trainer.randaugment_transform:
            return
        
        randaugment_transform = self.custom_progressive_trainer.randaugment_transform
        novel_metadata = self.custom_progressive_trainer.novel_metadata
        
        for i, im_file in enumerate(batch["im_file"]):
            if novel_metadata.get(Path(im_file).resolve().as_posix(), False):
                pil_image = to_pil_image(batch["img"][i].cpu())
                augmented_pil = randaugment_transform(pil_image)
                batch["img"][i] = to_tensor(augmented_pil).to(self.device)

    def on_fit_epoch_end(self, epoch):
        """Run validation at the configured `val_interval` boundary."""
        if RANK not in (-1, 0) or not self.custom_progressive_trainer: return
        val_interval = self.custom_progressive_trainer.config.get("val_interval", 5)
        if self.epoch > 0 and self.epoch % val_interval == 0:
            logger.info(f"📈 Triggering validation for epoch {self.epoch}...")
            self.run_validation()

    def on_val_end(self):
        """Stop the current phase early if mAP@0.5 exceeds the threshold."""
        if RANK not in (-1, 0) or not self.custom_progressive_trainer or not self.metrics: return
        current_map50 = self.metrics.get("metrics/mAP50(B)", 0.0)
        threshold = self.custom_progressive_trainer.config.get("phase_transition_mAP_threshold", 0.75)
        if current_map50 > threshold:
            logger.info(f"📈 Plateau reached! mAP@0.5 ({current_map50:.4f}) > threshold ({threshold}).")
            self.stop_training = True


class ProgressiveTrainer:
    """Coordinator for multi-phase training and custom evaluation workflow."""
    def __init__(self, config: Dict[str, Any], data_manager: DatasetManager, model_manager: ModelManager):
        self.config = config
        self.dm = data_manager
        self.mm = model_manager
        self.model = self.mm.create_model()
        self.novel_metadata = {}
        self.randaugment_transform = None
        self.full_history_df = pd.DataFrame()
        
        # Initialize the evaluator for use within the training loop
        self.evaluator = Evaluator(self.config, self.dm)
        # Initialize attributes for our custom "Safety Net" selection logic
        self.best_novel_map = -1.0
        self.overall_best_model_path = None

    def run_training(self):
        """Execute the full progressive training flow across all mixing phases."""
        logger.info("🚀 Starting progressive training workflow with custom evaluation...")
        mixing_phases = self.config.get("mixing_phases", [])

        # Mapping from our clean config keys to the Ultralytics trainer's expected argument keys
        ULTRALYTICS_ARGS = {
            "model_name": "model", "max_phase_epochs": "epochs", "phase_patience": "patience", "batch": "batch", 
            "imgsz": "imgsz", "device": "device", "optimizer": "optimizer", "lr0": "lr0", "lrf": "lrf", "cos_lr": "cos_lr", 
            "momentum": "momentum", "weight_decay": "weight_decay", "warmup_epochs": "warmup_epochs", 
            "warmup_momentum": "warmup_momentum", "warmup_bias_lr": "warmup_bias_lr", "box": "box", "cls": "cls", 
            "dfl": "dfl", "mosaic": "mosaic", "mixup": "mixup", "copy_paste": "copy_paste", "hsv_h": "hsv_h", 
            "hsv_s": "hsv_s", "hsv_v": "hsv_v", "degrees": "degrees", "translate": "translate", "scale": "scale", 
            "shear": "shear", "perspective": "perspective", "flipud": "flipud", "fliplr": "fliplr", "workers": "workers", 
            "amp": "amp", "cache": "cache", "project": "project", "name": "name", "plots": "plots", "save": "save", 
            "save_period": "save_period",
        }

        for i, phase_info in enumerate(mixing_phases):
            phase_num = i + 1
            logger.info(f"\n{'='*80}\n🚀 Starting Phase {phase_num}/{len(mixing_phases)}\n{'='*80}")
            
            try:
                # 1. Prepare data and augmentations for the current phase
                phase_yaml_path = self.dm.create_phase_training_yaml(phase_info.get("novel_ratio", 0.0))
                self._prepare_phase_metadata_and_augs(phase_yaml_path)

                # 2. Configure and create the YOLO trainer for this phase
                train_args = self._build_trainer_args(phase_yaml_path, phase_num)
                trainer = ProgressiveCallbackTrainer(overrides=train_args)
                trainer.model = self.model.model
                trainer.custom_progressive_trainer = self
                
                # 3. Run the training for this phase
                trainer.train()
                
                # 4. Process the results using our custom evaluation logic
                if RANK in (-1, 0):
                    self._process_phase_results(trainer, phase_num)
            
            except Exception as e:
                # On a critical failure (like OOM), log, finalize, and halt the entire script.
                logger.error(f"❌ Phase {phase_num} failed with a critical error: {e}", exc_info=True)
                logger.error("Halting the training workflow due to a non-recoverable error.")
                if RANK in (-1, 0): self._finalize_training()
                raise e # Re-raise the exception to stop the notebook/script
        
        if RANK in (-1, 0):
            self._finalize_training()
        return self.model

    def _prepare_phase_metadata_and_augs(self, phase_yaml_path: Path):
        """Load novel-image flags and prepare RandAugment for the current phase."""
        self.novel_metadata.clear()
        with open(phase_yaml_path, "r") as f: data_config = yaml.safe_load(f)
        metadata_path = Path(data_config.get("novel_metadata"))
        if metadata_path.exists():
            for line in metadata_path.read_text().splitlines():
                if "," in line:
                    path_str, is_novel_str = line.strip().split(",", 1)
                    self.novel_metadata[Path(path_str).resolve().as_posix()] = is_novel_str.lower() == "true"
        
        ra_config = self.config.get("randaugment", {})
        if ra_config.get("enabled", False):
            self.randaugment_transform = create_rand_augment_transform(ra_config)
        else:
            self.randaugment_transform = None

    def _build_trainer_args(self, phase_yaml_path: Path, phase_num: int) -> Dict[str, Any]:
        """Build the overrides dict consumed by Ultralytics `DetectionTrainer`."""
        train_args = {"data": str(phase_yaml_path)}
        # This mapping can be defined as a class constant if preferred
        ULTRALYTICS_ARGS = {
            "model_name": "model", "max_phase_epochs": "epochs", "phase_patience": "patience", "batch": "batch", 
            "imgsz": "imgsz", "device": "device", "optimizer": "optimizer", "lr0": "lr0", "lrf": "lrf", "cos_lr": "cos_lr", 
            "momentum": "momentum", "weight_decay": "weight_decay", "warmup_epochs": "warmup_epochs", 
            "warmup_momentum": "warmup_momentum", "warmup_bias_lr": "warmup_bias_lr", "box": "box", "cls": "cls", 
            "dfl": "dfl", "mosaic": "mosaic", "mixup": "mixup", "copy_paste": "copy_paste", "hsv_h": "hsv_h", 
            "hsv_s": "hsv_s", "hsv_v": "hsv_v", "degrees": "degrees", "translate": "translate", "scale": "scale", 
            "shear": "shear", "perspective": "perspective", "flipud": "flipud", "fliplr": "fliplr", "workers": "workers", 
            "amp": "amp", "cache": "cache", "project": "project", "name": "name", "plots": "plots", "save": "save", 
            "save_period": "save_period",
        }
        for cfg_key, trn_key in ULTRALYTICS_ARGS.items():
            if cfg_key in self.config:
                train_args[trn_key] = self.config[cfg_key]
        
        train_args["name"] = f'{self.config.get("task_name", "task")}_phase_{phase_num}'
        train_args["project"] = str(Path(self.config.get("output_dir", "outputs")) / "runs")
        train_args["val"] = True
        return train_args

    def _process_phase_results(self, trainer, phase_num):
        """Evaluate the best checkpoint of the phase and apply safety-net logic."""
        logger.info(f"--- Processing results for Phase {phase_num} with custom evaluation ---")
        run_dir = Path(trainer.save_dir)
        phase_best_ckpt = run_dir / "weights" / "best.pt"

        # 1. Aggregate the training history CSV for the final plot
        history_csv = run_dir / "results.csv"
        if history_csv.exists():
            phase_df = pd.read_csv(history_csv)
            phase_df["phase"] = phase_num
            phase_df.attrs['save_dir'] = run_dir
            self.full_history_df = pd.concat([self.full_history_df, phase_df], ignore_index=True)
            self.full_history_df.attrs['save_dir'] = run_dir

        if not phase_best_ckpt.exists():
            logger.warning(f"No 'best.pt' found for Phase {phase_num}. Skipping evaluation.")
            return

        # 2. Run our custom evaluator on the best model from this phase
        logger.info(f"Loading best model from Phase {phase_num} for detailed evaluation...")
        phase_model = YOLO(str(phase_best_ckpt))
        metrics = self.evaluator.evaluate_model(phase_model)

        if not metrics:
            logger.warning(f"Evaluation failed for the model from Phase {phase_num}.")
            return

        base_map = metrics.get('base_classes', {}).get('mAP50', 0.0)
        novel_map = metrics.get('novel_classes', {}).get('mAP50', 0.0)
        
        # 3. Apply the "Constrained Optimization" (Safety Net) rule
        safety_net_threshold = self.config.get("evaluation", {}).get("safety_net_threshold", 0.40)
        
        logger.info(f"Phase {phase_num} Model Perf: Base mAP={base_map:.4f}, Novel mAP={novel_map:.4f}")
        
        if base_map >= safety_net_threshold:
            logger.info(f"✅ Base mAP ({base_map:.4f}) PASSED the safety net ({safety_net_threshold:.4f}).")
            if novel_map > self.best_novel_map:
                logger.info(f"🏆🏆🏆 NEW OVERALL BEST MODEL FOUND! 🏆🏆🏆")
                logger.info(f"Novel mAP improved from {self.best_novel_map:.4f} to {novel_map:.4f}")
                self.best_novel_map = novel_map
                self.overall_best_model_path = phase_best_ckpt
            else:
                logger.info(f"Novel mAP ({novel_map:.4f}) did not improve upon the best of {self.best_novel_map:.4f}.")
        else:
            logger.warning(f"❌ Base mAP ({base_map:.4f}) FAILED the safety net ({safety_net_threshold:.4f}). Model rejected.")

    def _finalize_training(self):
        """Persist overall best model, save history CSV, and render plots."""
        output_dir = Path(self.config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.overall_best_model_path:
            logger.info(f"🏆 Saving overall best model from {self.overall_best_model_path} to {output_dir / 'best_model.pt'}...")
            shutil.copy2(self.overall_best_model_path, output_dir / "best_model.pt")
            self.model = YOLO(output_dir / "best_model.pt")
        else:
            logger.warning("⚠️ No best model was found across all phases that met the custom criteria.")
            try: self.model.save(str(output_dir / "last_model.pt"))
            except Exception: logger.error("Could not save the last model state.")
            
        if not self.full_history_df.empty:
            history_path = output_dir / "full_training_history.csv"
            self.full_history_df.to_csv(history_path, index=False)
            logger.info(f"💾 Full training history saved to {history_path}")
            save_training_plots(self.full_history_df, output_dir, show=False, task_name=self.config.get("task_name"))
        else:
            logger.warning("⚠️ Training history is empty. No summary plot will be generated.")