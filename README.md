# YOLO Few-Shot Object Detection

A comprehensive pipeline for few-shot object detection using YOLO models with **two-stage progressive training** and intelligent curriculum learning. This project enables training YOLO models on limited data samples (10-shot and 20-shot scenarios) while maintaining high detection accuracy through advanced training strategies.

## 🎯 Overview

This repository implements a sophisticated few-shot learning pipeline that uses:
- **Two-Stage Training Architecture**: Head training followed by full model fine-tuning
- **Progressive Curriculum Learning**: Gradual introduction of novel classes with configurable mixing ratios
- **Multi-Dataset Integration**: COCO (80 base classes) + HomeObjects (5 novel classes)
- **Advanced Layer Freezing**: Strategic backbone/neck freezing for optimal transfer learning
- **RandAugment Integration**: Advanced augmentation using timm library

### Proven Performance Results

**10-Shot Learning (14 images per class):**
| Metric | Base Classes | Novel Classes | Overall |
|--------|--------------|---------------|---------|
| mAP@50 | 47.40% | 26.65% | 46.17% |
| mAP@50-95 | 32.90% | 22.43% | 32.28% |

**20-Shot Learning (29 images per class):**
| Metric | Base Classes | Novel Classes | Overall |
|--------|--------------|---------------|---------|
| mAP@50 | 51.37% | 37.92% | 50.58% |
| mAP@50-95 | 36.74% | 29.77% | 36.33% |

### Key Features

- 🔄 **Two-Stage Progressive Training**: Stage 1 (head training) → Stage 2 (fine-tuning)
- 📊 **Dual Shot Scenarios**: 10-shot (14 images per class) and 20-shot (29 images per class)
- 🎨 **Advanced Augmentation**: RandAugment with mosaic and mixup
- 📈 **Comprehensive Evaluation**: Base vs novel class performance analysis
- 🔧 **Hierarchical Configuration**: Base config + stage-specific overrides
- 🚀 **Production Ready**: Modular architecture with Hugging Face Hub integration

## 🏗️ Architecture

The project contains two parallel pipelines optimized for different shot scenarios:

```
├── yolo_10_shot_detection/     # 10-shot learning pipeline (14 images per class)
│   ├── configs/
│   │   ├── base_config.yaml    # Global defaults for 10-shot
│   │   ├── task2_stage1.yaml   # Stage 1: Head training config
│   │   └── task2_stage2.yaml   # Stage 2: Fine-tuning config
│   └── src/                    # Core pipeline modules
└── yolo_20_shot_detection/     # 20-shot learning pipeline (29 images per class)
    ├── configs/
    │   ├── base_config.yaml    # Global defaults for 20-shot  
    │   ├── task2_stage1.yaml   # Stage 1: Head training config
    │   └── task2_stage2.yaml   # Stage 2: Fine-tuning config
    └── src/                    # Core pipeline modules
```

### Core Components

```
src/
├── data/
│   ├── dataset_manager.py      # HF Hub integration & dataset orchestration
│   └── augmentation.py         # RandAugment transforms via timm
├── training/
│   ├── progressive_trainer.py  # Multi-phase progressive training
│   └── model_manager.py        # Model lifecycle & layer freezing
├── evaluation/
│   └── evaluator.py           # Base vs novel class evaluation
├── visualization/
│   └── plotter.py             # Training plots & inference visualization  
└── utils/
    └── config_validator.py     # Configuration merging & validation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics torch torchvision huggingface_hub pyyaml matplotlib seaborn timm
```

### Installation

```bash
git clone https://github.com/VivekChandra1324/yolo-few-shot-object-detection.git
cd yolo-few-shot-object-detection

# Choose your shot scenario
cd yolo_10_shot_detection  # or yolo_20_shot_detection
```

### Quick Start

```python
from src.utils.config_validator import load_and_merge_configs
from src.data.dataset_manager import DatasetManager
from src.training.progressive_trainer import ProgressiveTrainer
from src.training.model_manager import ModelManager

# Load hierarchical configuration (base + stage)
config = load_and_merge_configs("configs/base_config.yaml", "configs/task2_stage1.yaml")

# Initialize components
dataset_manager = DatasetManager(config)
model_manager = ModelManager(config, dataset_manager)
trainer = ProgressiveTrainer(config, dataset_manager, model_manager)

# Setup datasets and train
dataset_manager.setup_datasets("task2")  
trainer.run_training()
```

## 📊 Dataset Configuration

### Supported Datasets

The pipeline integrates datasets from Hugging Face Hub:

**10-Shot Configuration:**
- **COCO**: `VivekChandra/COCO_143` - 80 base object classes
- **HomeObjects**: `VivekChandra/HomeObjects-3K_5class_14_per_class` - 5 novel classes, 14 images each

**20-Shot Configuration:** 
- **COCO**: `VivekChandra/COCO_143` - 80 base object classes
- **HomeObjects**: `VivekChandra/HomeObjects-3K_5class_29_per_class` - 5 novel classes, 29 images each

### Dataset Split Strategy

**10-Shot (14 total images per class):**
- Train: 10 shots
- Validation: 3 shots  
- Test: 1 shot

**20-Shot (29 total images per class):**
- Train: 20 shots
- Validation: 6 shots
- Test: 3 shots

### Automatic Data Pipeline

The `DatasetManager` handles:
- ✅ HF Hub dataset downloading and extraction
- ✅ Label remapping (novel classes follow base classes: 0-79 → base, 80-84 → novel)
- ✅ YOLO format validation and conversion
- ✅ Progressive phase training manifest generation

## 🎓 Two-Stage Training Pipeline

### Stage Architecture

**Stage 1: Detection Head Training**
- **Purpose**: Train detection head with backbone heavily frozen
- **Model**: Start from `yolo11n.pt` 
- **Learning Rate**: 0.001
- **Max Epochs**: 75

**Stage 2: Full Model Fine-tuning**
- **Purpose**: Unfreeze more layers and fine-tune with low LR
- **Model**: Resume from Stage 1 best checkpoint
- **Learning Rate**: 0.000067 (scaled for batch size 512)
- **Max Epochs**: 100

### Progressive Mixing Phases

Both stages use progressive curriculum learning with 6 phases:

```yaml
mixing_phases:
  - {novel_ratio: 0.05}    # 5% novel images
  - {novel_ratio: 0.1}     # 10% novel images
  - {novel_ratio: 0.25}    # 25% novel images
  - {novel_ratio: 0.5}     # 50% novel images
  - {novel_ratio: 0.75}    # 75% novel images
  - {novel_ratio: 0.9}     # 90% novel images
```

### Key Training Parameters

```yaml
# Global settings (base_config.yaml)
num_classes: 85           # 80 base + 5 novel
device: "0,1"            # Multi-GPU support
batch: 512               # Global batch size
optimizer: "AdamW"       
epochs: 75               # Per-stage limit
patience: 30             # Early stopping
phase_patience: 40       # Phase transition patience
phase_transition_mAP_threshold: 0.65
```

## 📈 Advanced Features

### RandAugment Integration

Uses `timm` library for advanced augmentation:

```yaml
randaugment:
  enabled: true
  num_ops: 2              # Number of operations per image
  magnitude: 7            # Augmentation strength
  mstd: 0.4              # Magnitude standard deviation
```

Additional augmentations:
- **Mosaic**: 0.25 probability
- **Mixup**: 0.3 probability  
- All other built-in augmentations disabled for reproducibility

### Model Management

- **Strategic Freezing**: Different freeze depths per stage
- **Checkpoint Recovery**: Automatic best model selection
- **Parameter Analysis**: Trainable vs frozen parameter logging

## ⚙️ Configuration System

### Hierarchical Configuration

The system uses a three-level hierarchy:

1. **Base Config** (`base_config.yaml`): Global defaults for each shot scenario
2. **Stage Config** (`task2_stage1.yaml`, `task2_stage2.yaml`): Stage-specific overrides
3. **Runtime Merging**: Configs merged at runtime using `load_and_merge_configs()`

### 10-Shot Example

**Base Configuration:**
```yaml
# base_config.yaml
seed: 42
num_classes: 85
batch: 512
dataset:
  hf_repos:
    homeobjects_14:  # 10-shot dataset
      repo_id: "VivekChandra/HomeObjects-3K_5class_14_per_class"
  homeobjects:
    train_shots: 10
    val_shots: 3
    test_shots: 1
```

**Stage 1 Override:**
```yaml
# task2_stage1.yaml
task_name: "task_10shot_stage1_train_head"
model_name: "yolo11n.pt"
freeze: 22
lr0: 0.001
max_phase_epochs: 75
```

## 📝 Usage Examples

### Complete Two-Stage Training

```python
from src.utils.config_validator import load_and_merge_configs
from src.data.dataset_manager import DatasetManager
from src.training.progressive_trainer import ProgressiveTrainer
from src.training.model_manager import ModelManager

# Stage 1: Head Training
stage1_config = load_and_merge_configs(
    "configs/base_config.yaml", 
    "configs/task2_stage1.yaml"
)

dataset_manager = DatasetManager(stage1_config)
dataset_manager.setup_datasets("task2")

model_manager = ModelManager(stage1_config, dataset_manager)
trainer = ProgressiveTrainer(stage1_config, dataset_manager, model_manager)

# Train Stage 1
stage1_best = trainer.run_training()

# Stage 2: Fine-tuning
stage2_config = load_and_merge_configs(
    "configs/base_config.yaml", 
    "configs/task2_stage2.yaml"
)
# Update model path to Stage 1 best
stage2_config['model_name'] = trainer.overall_best_model_path

# Train Stage 2
trainer2 = ProgressiveTrainer(stage2_config, dataset_manager, model_manager)
final_model = trainer2.run_training()
```

### Custom Evaluation

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(config, dataset_manager)
results = evaluator.evaluate_model(final_model)

print(f"Overall mAP@0.5: {results['overall']['mAP50']:.3f}")
print(f"Base Classes mAP: {results['base_classes']['mAP50']:.3f}")
print(f"Novel Classes mAP: {results['novel_classes']['mAP50']:.3f}")
```

### Visualization

```python
from src.visualization.plotter import view_sample_images, save_training_plots, visualize_inference

# View sample images (requires dataset object, not dataset_manager)
# view_sample_images(dataset, num_samples=4, class_names=class_names)

# Training progress plots
save_training_plots(trainer.full_history_df, "outputs/plots", task_name="my_experiment")

# Inference visualization
visualize_inference(model, test_images, num_samples=4, conf_thresh=0.25)
```

## 🔧 Advanced Configuration

### Learning Rate Scaling

```python
from src.utils.config_validator import calculate_scaled_learning_rate

# Automatic LR scaling based on batch size
scaled_lr = calculate_scaled_learning_rate(
    base_lr=0.001,
    base_batch_size=64,
    new_batch_size=512
)
```

### Loss Weight Optimization

```yaml
# Optimized for few-shot scenarios
box: 7.5      # Bounding box loss weight  
cls: 1.5      # Classification loss weight
dfl: 1.5      # Distribution focal loss weight
```

## 🛠️ Implementation Notes

### Task Implementation

- **Dataset Key Mapping**: 
  - 10-shot uses `homeobjects_14` key
  - 20-shot uses `homeobjects_29` key
- **Class Structure**: 85 total classes (80 COCO base + 5 HomeObjects novel)

### Configuration Validation

The system includes comprehensive validation:

```python
from src.utils.config_validator import validate_base_config, validate_config_file

# Validate configuration
errors = validate_base_config(config)
if errors:
    print("Configuration errors:", errors)
```

### Safety Net Evaluation

The system implements a "safety net" to ensure base class performance doesn't degrade:

- **Threshold**: 0.40 mAP@0.5 for base classes
- **Selection Logic**: Only models that pass the safety net are considered for final selection
- **Best Model**: Selected based on highest novel class mAP while maintaining base performance

## 📊 Training Outputs

The pipeline generates several outputs:

- **`best_model.pt`**: Final model that passed safety net evaluation
- **`full_training_history.csv`**: Complete training metrics across all phases
- **`training_summary_plots_*.png`**: Visualization of training progress
- **Phase-specific checkpoints**: Individual model weights for each progressive phase


## 🙏 Acknowledgments

- Ultralytics YOLO for the YOLO implementation
- Hugging Face Hub for dataset hosting
- timm for RandAugment implementation

---

**Author**: Vivek Chandra  
**Repository**: [VivekChandra1324/yolo-few-shot-object-detection](https://github.com/VivekChandra1324/yolo-few-shot-object-detection)
