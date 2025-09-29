# src/visualization/plotter.py

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from PIL import Image

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def view_sample_images(dataset, num_samples: int = 4, class_names: dict = None):
    """
    Display a grid of random sample images with ground-truth boxes.

    Useful for quickly verifying dataset integrity and annotations.
    """
    if not class_names:
        logger.warning("No class_names provided; labels will be displayed as numeric IDs.")
        class_names = {}

    n_cols = min(num_samples, 2)
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows))
    axes = axes.flatten()
    
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

    for i, ax_idx in enumerate(range(len(indices))):
        ax = axes[ax_idx]
        image_tensor, labels = dataset[indices[i]]
        
        img = image_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        h, w, _ = img.shape
        
        for box in labels:
            class_id, cx, cy, bw, bh = box.tolist()
            
            x_min = (cx - bw / 2) * w
            y_min = (cy - bh / 2) * h
            box_width = bw * w
            box_height = bh * h
            
            rect = patches.Rectangle(
                (x_min, y_min), box_width, box_height,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
            
            class_name = class_names.get(int(class_id), f"ID:{int(class_id)}")
            
            # --- SUGGESTION 1: Improved text positioning to prevent labels from going off-screen ---
            text_y_pos = max(y_min - 5, 15) # Ensures text is not drawn at y < 15
            ax.text(
                x_min, text_y_pos, class_name, color='white',
                fontsize=12, bbox=dict(facecolor='cyan', alpha=0.8, pad=1)
            )

        ax.set_title(f"Sample Index: {indices[i]}")
        ax.axis('off')
    
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def save_training_plots(history_df: pd.DataFrame, output_dir: Path, show: bool = True, task_name: str = None):
    """
    Generate and save summary plots from the training history DataFrame.

    The function is resilient to minor column-name variations.
    """
    if history_df.empty:
        logger.warning("Training history DataFrame is empty. Skipping plot generation.")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create task-specific filename to avoid overwrites
        plot_filename = f'training_summary_plots_{task_name}.png' if task_name else 'training_summary_plots.png'
        plot_path = output_dir / plot_filename
        logger.info(f"Available columns for plotting: {list(history_df.columns)}")

        train_loss_col = next((col for col in history_df.columns if 'train' in col and 'box_loss' in col), None)
        val_loss_col = next((col for col in history_df.columns if 'val' in col and 'box_loss' in col), None)
        map50_col = next((col for col in history_df.columns if 'map50' in col.lower() and '95' not in col.lower()), None)
        map50_95_col = next((col for col in history_df.columns if 'map50-95' in col.lower() or 'map50_95' in col.lower()), None)
        
        # --- SUGGESTION 2: Simplified and more robust pattern for finding the learning rate column ---
        lr_col = next((col for col in history_df.columns if 'lr' in col.lower()), None)
        
        # --- SUGGESTION 3: Slightly wider figure for better readability ---
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Training & Validation Metrics', fontsize=18)

        # Helper to safely plot a single column if present and numeric
        def plot_line(ax, df: pd.DataFrame, ycol: str, label: str = None, color: str = None):
            if not ycol or ycol not in df.columns:
                return False
            series = pd.to_numeric(df[ycol], errors='coerce')
            if series.notna().sum() == 0:
                logger.warning(f"Column '{ycol}' is non-numeric or empty after coercion; skipping plot.")
                return False
            if color:
                sns.lineplot(data=df, x=df.index, y=series, ax=ax, label=label, color=color)
            else:
                sns.lineplot(data=df, x=df.index, y=series, ax=ax, label=label)
            return True
        
        # Plot 1: Box Loss
        ax = axes[0, 0]
        has_train = plot_line(ax, history_df, train_loss_col, label='Train Box Loss')
        has_val = plot_line(ax, history_df, val_loss_col, label='Val Box Loss')
        ax.set_title('Box Loss vs. Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        if has_train or has_val:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Loss data not available', ha='center')

        # Plot 2: mAP@0.5
        ax = axes[0, 1]
        has_map50 = plot_line(ax, history_df, map50_col, color='g')
        ax.set_title('Validation mAP@0.5 vs. Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5')
        if not has_map50:
            ax.text(0.5, 0.5, 'mAP@0.5 data not available', ha='center')

        # Plot 3: mAP@0.5:0.95
        ax = axes[1, 0]
        has_map95 = plot_line(ax, history_df, map50_95_col, color='r')
        ax.set_title('Validation mAP@0.5:0.95 vs. Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP@0.5:0.95')
        if not has_map95:
            ax.text(0.5, 0.5, 'mAP@0.5:0.95 data not available', ha='center')
        
        # Plot 4: Learning Rate
        ax = axes[1, 1]
        has_lr = plot_line(ax, history_df, lr_col, color='purple')
        ax.set_title('Learning Rate Schedule vs. Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        if not has_lr:
            ax.text(0.5, 0.5, 'LR data not available', ha='center')

        if 'phase' in history_df.columns:
            phase_change_epochs = history_df.drop_duplicates('phase', keep='first').index
            if len(phase_change_epochs) > 1:
                for ax_row in axes:
                    for ax_col in ax_row:
                        for epoch in phase_change_epochs[1:]:
                            ax_col.axvline(x=epoch, color='gray', linestyle='--', alpha=0.8, label='Phase Change' if epoch == phase_change_epochs[1] else "")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        logger.info(f"✅ Training summary plots saved to {plot_path}")
        if show:
            plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"Could not generate or save plots due to an error: {e}")


def visualize_inference(model, image_paths: list, num_samples: int = 4, conf_thresh: float = 0.25):
    """
    Run inference on sample images and display detections.
    """
    if not image_paths:
        logger.warning("No image paths provided for inference visualization.")
        return
        
    sample_paths = random.sample(image_paths, k=min(num_samples, len(image_paths)))
    logger.info(f"Running inference on {len(sample_paths)} sample images...")
    
    # For multi-GPU setups, ensure batch size is compatible
    batch_size = 2 if len(sample_paths) >= 2 else 1
    results = model.predict(source=sample_paths, conf=conf_thresh, batch=batch_size)
    
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(im)
        ax.set_title(f"Inference Result: {Path(r.path).name}")
        ax.axis('off')
        plt.show()