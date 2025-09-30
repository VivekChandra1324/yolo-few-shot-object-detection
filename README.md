# YOLO Few-Shot Object Detection

A comprehensive pipeline for few-shot object detection using YOLO models with progressive curriculum learning and intelligent phase transitions. This project enables training YOLO models on limited data samples (10-shot and 20-shot scenarios) while maintaining high detection accuracy through advanced training strategies.

## 🎯 Overview

This repository implements a state-of-the-art few-shot learning pipeline that combines:
- **Progressive Curriculum Learning**: Gradual introduction of novel classes during training
- **Intelligent Phase Transitions**: Adaptive training phases based on performance metrics
- **Multi-Dataset Integration**: Seamless combination of COCO and HomeObjects datasets
- **Automated Data Management**: End-to-end dataset preparation and validation

### Key Features

- 🔄 **Progressive Training**: Multi-phase training with intelligent curriculum learning
- 📊 **Dual Shot Scenarios**: Support for both 10-shot and 20-shot detection tasks
- 🎨 **Advanced Augmentation**: RandAugment integration for improved generalization
- 📈 **Comprehensive Evaluation**: Built-in metrics and visualization tools
- 🔧 **Flexible Configuration**: YAML-based configuration system
- 🚀 **Production Ready**: Modular architecture with extensive error handling

## 🏗️ Architecture

The project is structured into two main variants:
- `yolo_10_shot_detection/`: Pipeline optimized for 10-shot learning scenarios
- `yolo_20_shot_detection/`: Pipeline optimized for 20-shot learning scenarios

### Core Components

```
src/
├── data/                    # Data management and preprocessing
│   ├── dataset_manager.py   # Dataset orchestration and HF Hub integration
│   └── augmentation.py      # RandAugment transformations
├── training/                # Training pipeline
│   ├── progressive_trainer.py  # Multi-phase progressive training
│   └── model_manager.py     # Model lifecycle management
├── evaluation/              # Evaluation and metrics
│   └── evaluator.py         # Comprehensive evaluation suite
├── visualization/           # Plotting and visualization
│   └── plotter.py          # Training plots and inference visualization
└── utils/                   # Utilities and configuration
    └── config_validator.py  # Configuration validation and setup
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics torch torchvision huggingface_hub pyyaml matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VivekChandra1324/yolo-few-shot-object-detection.git
cd yolo-few-shot-object-detection
```

2. Choose your shot scenario:
```bash
# For 10-shot detection
cd yolo_10_shot_detection

# For 20-shot detection  
cd yolo_20_shot_detection
```

3. Configure your training parameters in the YAML configuration files

### Quick Start

```python
from src import DatasetManager, ProgressiveTrainer, ModelManager, Evaluator

# Initialize components
config = load_and_validate_config("config.yaml")
dataset_manager = DatasetManager(config)
model_manager = ModelManager(config, dataset_manager)
trainer = ProgressiveTrainer(config, dataset_manager, model_manager)

# Setup datasets
dataset_manager.setup_datasets("task1")  # or "task2"

# Train with progressive curriculum
trainer.train_progressive()

# Evaluate results
evaluator = Evaluator(config, dataset_manager)
results = evaluator.evaluate_model(trainer.best_model_path)
```

## 📊 Dataset Management

### Supported Datasets

The pipeline integrates multiple datasets from Hugging Face Hub:

- **COCO**: Base dataset with 80 object classes
- **HomeObjects**: Novel object classes (143 classes for task1, 29 classes for task2)

### Dataset Configuration

```yaml
dataset:
  hf_repos:
    coco_143:
      repo_id: "your-org/coco-dataset"
      filename: "coco_yolo_format.zip"
    homeobjects_143:
      repo_id: "your-org/homeobjects-143"
      filename: "homeobjects_143.zip"
    homeobjects_29:
      repo_id: "your-org/homeobjects-29" 
      filename: "homeobjects_29.zip"
```

### Automated Data Pipeline

The `DatasetManager` handles:
- ✅ Automatic dataset downloading from Hugging Face Hub
- ✅ Label remapping for consistent class indices
- ✅ YOLO format validation and conversion
- ✅ Train/validation/test split management
- ✅ Phase-specific training manifest generation

## 🎓 Training Pipeline

### Progressive Curriculum Learning

The training pipeline implements a sophisticated multi-phase approach:

1. **Phase 1**: Base classes only (COCO dataset)
2. **Phase 2**: Gradual introduction of novel classes
3. **Phase 3**: Full dataset with all classes
4. **Phase 4**: Fine-tuning and optimization

### Key Training Features

- **Adaptive Learning Rates**: Automatic scaling based on dataset size
- **Layer Freezing**: Strategic backbone freezing for transfer learning  
- **Performance Monitoring**: Intelligent phase transitions based on metrics
- **Checkpoint Management**: Automatic best model saving and recovery

### Training Configuration

```yaml
training:
  model: "yolov8n.pt"
  epochs_per_phase: [50, 75, 100, 50]
  learning_rate: 0.001
  batch_size: 16
  freeze_layers: 10
  early_stopping_patience: 15
```

## 📈 Evaluation and Metrics

### Comprehensive Evaluation Suite

The `Evaluator` class provides:
- **mAP Calculations**: mAP@0.5 and mAP@0.5:0.95
- **Class-wise Analysis**: Per-class performance breakdown
- **Novel vs Base Performance**: Specialized few-shot metrics
- **Visualization Tools**: Confusion matrices, precision-recall curves

### Sample Evaluation

```python
evaluator = Evaluator(config, dataset_manager)
results = evaluator.evaluate_model(model_path)

print(f"Overall mAP@0.5: {results['map50']:.3f}")
print(f"Novel Classes mAP: {results['novel_map']:.3f}")
print(f"Base Classes mAP: {results['base_map']:.3f}")
```

## 🎨 Visualization Tools

### Built-in Visualization

- **Training Progress**: Loss curves, mAP progression, learning rate schedules
- **Sample Predictions**: Annotated inference results
- **Dataset Samples**: Augmented training samples with labels

```python
from src.visualization import view_sample_images, save_training_plots, visualize_inference

# View augmented training samples
view_sample_images(dataset_manager, num_samples=9)

# Generate training plots
save_training_plots(trainer.metrics_history, output_dir)

# Visualize model predictions
visualize_inference(model, test_images, class_names)
```

## ⚙️ Configuration System

### Flexible YAML Configuration

The pipeline uses a hierarchical configuration system:

```yaml
# Global settings
seed: 42
output_dir: "outputs"
device: "0"  # GPU device

# Dataset configuration
dataset:
  hf_repos: {...}

# Training configuration  
training:
  model: "yolov8n.pt"
  progressive_phases:
    phase1: {novel_ratio: 0.0, epochs: 50}
    phase2: {novel_ratio: 0.3, epochs: 75}
    phase3: {novel_ratio: 0.7, epochs: 100}
    phase4: {novel_ratio: 1.0, epochs: 50}

# Evaluation settings
evaluation:
  confidence_threshold: 0.25
  iou_threshold: 0.45
```

### Configuration Validation

Automatic validation ensures:
- ✅ Required parameters are present
- ✅ Value ranges are appropriate
- ✅ Dataset repositories are accessible
- ✅ Model configurations are valid

## 📝 Usage Examples

### Basic Training Pipeline

```python
from src import load_and_validate_config, DatasetManager, ProgressiveTrainer, ModelManager

# Load and validate configuration
config = load_and_validate_config("configs/base_config.yaml")

# Initialize pipeline components
dataset_manager = DatasetManager(config)
model_manager = ModelManager(config, dataset_manager)
trainer = ProgressiveTrainer(config, dataset_manager, model_manager)

# Setup and train
dataset_manager.setup_datasets("task1")
trainer.train_progressive()
```

### Custom Evaluation

```python
from src.evaluation import Evaluator

evaluator = Evaluator(config, dataset_manager)
results = evaluator.evaluate_model("outputs/models/best.pt")

# Class-wise performance analysis
for class_name, ap in results['class_ap'].items():
    print(f"{class_name}: {ap:.3f}")
```

### Advanced Configuration

```python
from src.utils import calculate_scaled_learning_rate

# Calculate optimal learning rate based on dataset size
optimal_lr = calculate_scaled_learning_rate(
    base_lr=0.001,
    dataset_size=len(dataset_manager.novel_train_filenames)
)
```

## 🔧 Advanced Features

### Model Management
- **Multi-GPU Support**: Automatic device detection and optimization
- **Memory Optimization**: Efficient batch processing and caching
- **Checkpoint Recovery**: Resume training from any phase
- **Model Introspection**: Parameter counting and layer analysis

### Data Augmentation
- **RandAugment Integration**: Advanced augmentation policies
- **Progressive Augmentation**: Phase-specific augmentation strategies
- **Reproducible Augmentation**: Seed-controlled transformations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLO implementation
- **Hugging Face**: For dataset hosting and model hub integration
- **COCO Dataset**: For providing the base object detection dataset
- **HomeObjects**: For novel object class annotations

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review existing discussions and solutions

---

**Author**: YOLO Few-Shot Learning Team  
**Version**: 1.0.0  
**Last Updated**: 2024
