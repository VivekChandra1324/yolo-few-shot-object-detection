# src/data/dataset_manager.py

import logging
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

# --- Setup Logging ---
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    End-to-end dataset orchestration for few-shot detection.

    Responsibilities:
    - Download and extract source datasets from Hugging Face Hub (base + novel).
    - Remap novel class indices to follow base classes (contiguous label space).
    - Materialize a combined YOLO-format dataset on disk (train/val/test).
    - Generate phase-specific training manifests for progressive mixing.
    - Persist a master `dataset.yaml` consumed by Ultralytics.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration and derive working directories."""
        self.config = config
        self.seed = self.config.get("seed", 42)
        self._set_seeds()

        # --- Core Path Definitions ---
        self.output_dir = Path(self.config.get("output_dir", "outputs"))
        self.dataset_root = self.output_dir / "datasets"
        self.coco_source_dir = self.dataset_root / "coco_source"
        self.homeobjects_source_dir = self.dataset_root / "homeobjects_source"
        self.combined_data_root: Path = Path()

        # --- State Attributes ---
        self.base_classes: List[str] = []
        self.novel_classes: List[str] = []
        self.num_total_classes = 0
        self.coco_yaml_path: Path = Path()
        self.novel_yaml_path: Path = Path()
        self.master_yaml_path: Path = Path()
        self.novel_train_filenames: Set[str] = set()

    def _set_seeds(self):
        """Sets random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        logger.info(f"🌱 Random seeds set to {self.seed} for reproducibility.")

    def setup_datasets(self, task_type: str) -> None:
        """Prepare combined dataset (idempotent after first run)."""
        logger.info(f"🚀 Starting dataset setup for task: '{task_type}'...")
        self.combined_data_root = self.output_dir / "combined_dataset" / f"{task_type}"

        master_yaml_path = self.combined_data_root / "dataset.yaml"
        if master_yaml_path.exists():
            logger.info(f"📁 Existing dataset found at {self.combined_data_root}. Loading configuration...")
            self._load_existing_dataset(task_type)
            return

        logger.info("✨ Performing first-time setup for the dataset.")
        self.base_classes, self.coco_yaml_path = self._prepare_source_dataset("coco_143", self.coco_source_dir)
        
        novel_repo_key = "homeobjects_143" if task_type == "task1" else "homeobjects_14"
        self.novel_classes, self.novel_yaml_path = self._prepare_source_dataset(
            novel_repo_key, self.homeobjects_source_dir / task_type
        )

        novel_train_dir = self.novel_yaml_path.parent / "images" / "train"  # [PATH CORRECTION]
        
        logger.info(f"Scanning for novel training images in: {novel_train_dir}")
        if not novel_train_dir.is_dir():
            logger.error(f"FATAL: Source directory for novel training images not found at the expected path.")
            self.novel_train_filenames = set()
        else:
            self.novel_train_filenames = {p.name for p in novel_train_dir.glob("*.*")}
            logger.info(f"✅ Found {len(self.novel_train_filenames)} novel training image filenames.")

        if not self.novel_train_filenames:
            logger.warning("⚠️ The 'novel_train_filenames' set is empty. This will prevent filtering of base/novel images.")
        
        self.num_total_classes = len(self.base_classes) + len(self.novel_classes)
        
        self._create_combined_dataset_on_disk()
        self._create_master_yaml()
        logger.info("✅ First-time dataset setup complete.")

    def _load_existing_dataset(self, task_type: str) -> None:
        """Load metadata/paths from an already processed combined dataset."""
        master_yaml_path = self.combined_data_root / "dataset.yaml"
        with open(master_yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        self.num_total_classes = yaml_content['nc'] 
        class_names = list(yaml_content['names'].values())
        self.base_classes = class_names[:80]
        self.novel_classes = class_names[80:]
        self.master_yaml_path = master_yaml_path
        
        novel_repo_key = "homeobjects_143" if task_type == "task1" else "homeobjects_14"
        novel_yaml_files = list((self.homeobjects_source_dir / task_type).glob("**/dataset.yaml"))
        if novel_yaml_files:
            self.novel_yaml_path = novel_yaml_files[0]
            novel_train_dir = self.novel_yaml_path.parent / "images" / "train"  # [PATH CORRECTION]
            if novel_train_dir.exists():
                self.novel_train_filenames = {p.name for p in novel_train_dir.glob("*.*")}
        
        coco_yaml_files = list(self.coco_source_dir.glob("**/dataset.yaml"))
        if coco_yaml_files:
            self.coco_yaml_path = coco_yaml_files[0]
        
        logger.info(f"✅ Loaded existing dataset configuration. Found {len(self.novel_train_filenames)} novel train filenames.")

    def _prepare_source_dataset(self, repo_key: str, target_dir: Path) -> Tuple[List[str], Path]:
        """Download, extract, and parse a source dataset; return names and YAML path."""
        hf_config = self.config["dataset"]["hf_repos"][repo_key]
        self._download_and_extract(hf_config["repo_id"], hf_config["filename"], target_dir)
        return self._parse_dataset_yaml(target_dir)

    def _download_and_extract(self, repo_id: str, filename: str, target_dir: Path):
        """Fetch archive from HF Hub and extract it unless already present."""
        target_dir.mkdir(parents=True, exist_ok=True)
        if any(target_dir.glob("**/dataset.yaml")):
            logger.info(f"Dataset '{repo_id}' already found at {target_dir}. Skipping download.")
            return
        
        logger.info(f"Downloading '{filename}' from Hugging Face repo '{repo_id}'...")
        zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(target_dir))
        
        logger.info(f"Extracting '{filename}' to {target_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        
        os.remove(zip_path)
        logger.info(f"Extraction complete for '{repo_id}'.")

    def _parse_dataset_yaml(self, search_path: Path) -> Tuple[List[str], Path]:
        """Locate and parse `dataset.yaml`; return class names and YAML path."""
        try:
            yaml_path = next(search_path.glob("**/dataset.yaml"))
            with open(yaml_path, "r") as f: content = yaml.safe_load(f)
            return list(content["names"].values()), yaml_path
        except StopIteration:
            raise FileNotFoundError(f"dataset.yaml not found in the extracted files at {search_path}")

    def _create_combined_dataset_on_disk(self):
        """Construct a single YOLO dataset by copying base and remapped novel data."""
        logger.info("💿 Creating combined dataset on disk...")
        if self.combined_data_root.exists(): shutil.rmtree(self.combined_data_root)
        
        base_root = self.coco_yaml_path.parent
        novel_root = self.novel_yaml_path.parent
        num_base = len(self.base_classes)
        
        for split in ["train", "val", "test"]:
            logger.info(f"  - Processing '{split}' split...")
            img_dir = self.combined_data_root / split / "images"
            lbl_dir = self.combined_data_root / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in (base_root / "images" / split).glob("*.*"):
                if (base_root / "labels" / split / f"{img_file.stem}.txt").exists():
                    shutil.copy(img_file, img_dir)
                    shutil.copy(base_root / "labels" / split / f"{img_file.stem}.txt", lbl_dir)

            for img_file in (novel_root / "images" / split).glob("*.*"):
                label_file = novel_root / "labels" / split / f"{img_file.stem}.txt"
                if label_file.exists():
                    remapped_lines = [f"{int(p[0]) + num_base} {' '.join(p[1:])}" for p in [ln.strip().split() for ln in label_file.read_text().splitlines()] if len(p) >= 5]
                    if remapped_lines:
                        shutil.copy(img_file, img_dir)
                        (lbl_dir / label_file.name).write_text("\n".join(remapped_lines))

    def _create_master_yaml(self):
        """Write the master `dataset.yaml` consumed by Ultralytics."""
        self.master_yaml_path = self.combined_data_root / "dataset.yaml"
        all_class_names = self.base_classes + self.novel_classes
        yaml_content = {
            "path": str(self.combined_data_root.resolve()), "train": "train/images", "val": "val/images", 
            "test": "test/images", "nc": self.num_total_classes, "names": dict(enumerate(all_class_names)),
        }
        with open(self.master_yaml_path, "w") as f: yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
        logger.info(f"Master dataset YAML created at: {self.master_yaml_path}")

    def create_phase_training_yaml(self, novel_ratio: float) -> Path:
        """Create phase-specific train list, novel metadata, and a `data.yaml`."""
        logger.info(f"📝 Creating phase-specific YAML (novel_ratio={novel_ratio:.2f})...")
        rng = np.random.default_rng(self.seed)

        all_train_images = list((self.combined_data_root / "train" / "images").glob("*.*"))
        base_paths = np.array([p for p in all_train_images if p.name not in self.novel_train_filenames])
        novel_paths = np.array([p for p in all_train_images if p.name in self.novel_train_filenames])

        if not novel_paths.any(): logger.warning("No novel images found for sampling.")
        if not base_paths.any(): logger.warning("No base images found for sampling.")

        num_novel = min(int(round(novel_ratio * len(novel_paths))), len(novel_paths))
        num_base = min(int(round((1.0 - novel_ratio) * len(base_paths))), len(base_paths))

        sampled_novel = rng.choice(novel_paths, size=num_novel, replace=False) if num_novel > 0 else np.array([])
        sampled_base = rng.choice(base_paths, size=num_base, replace=False) if num_base > 0 else np.array([])
        
        final_image_paths = np.concatenate([sampled_base, sampled_novel]); rng.shuffle(final_image_paths)
        logger.info(f"Phase dataset created with {len(sampled_novel)} novel and {len(sampled_base)} base images (Total: {len(final_image_paths)}).")
        
        phase_dir = self.combined_data_root / "phase_definitions"; phase_dir.mkdir(exist_ok=True)
        phase_id = int(novel_ratio * 100)
        
        train_list_path = phase_dir / f"train_list_phase_{phase_id}.txt"
        metadata_path = phase_dir / f"novel_metadata_phase_{phase_id}.txt"
        yaml_path = phase_dir / f"data_phase_{phase_id}.yaml"

        train_list_path.write_text("\n".join([str(p.resolve()) for p in final_image_paths]))
        
        with open(metadata_path, "w") as f:
            for p in final_image_paths: f.write(f"{p.resolve().as_posix()},{p.name in self.novel_train_filenames}\n")
        
        yaml_content = {
            "path": str(self.combined_data_root.resolve()), "train": str(train_list_path.resolve()), "val": "val/images",
            "test": "test/images", "nc": self.num_total_classes, "names": dict(enumerate(self.base_classes + self.novel_classes)),
            "novel_metadata": str(metadata_path.resolve()),
        }
        with open(yaml_path, "w") as f: yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
            
        logger.info(f"✅ Phase YAML and metadata created: {yaml_path}")
        return yaml_path