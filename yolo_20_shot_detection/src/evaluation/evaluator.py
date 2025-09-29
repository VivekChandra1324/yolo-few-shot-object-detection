# src/evaluation/evaluator.py

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from src.data.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Final evaluation on the test set with base vs. novel breakdown.

    Computes overall metrics and splits results into base and novel class groups
    for clearer few-shot analysis.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DatasetManager):
        self.config = config
        self.dm = data_manager
        self.base_class_ids = list(range(len(self.dm.base_classes)))
        self.novel_class_ids = list(range(
            len(self.dm.base_classes),
            len(self.dm.base_classes) + len(self.dm.novel_classes)
        ))
        logger.info(
            f"Evaluator initialized for {len(self.base_class_ids)} base and "
            f"{len(self.novel_class_ids)} novel classes."
        )

    def evaluate_model(self, model: YOLO) -> Dict[str, Any]:
        """
        Run validation on the test split and return parsed metrics.
        """
        logger.info("🚀 Starting final model evaluation on the test set...")
        
        eval_output_dir = Path(self.config.get("output_dir", "outputs")) / "evaluation"
        run_name = f"final_evaluation_{self.config.get('task_name', 'run')}"
        
        try:
            metrics: DetMetrics = model.val(
                data=str(self.dm.master_yaml_path),
                split="test",
                device=self.config.get("validation_device", "0"),
                batch=self.config.get("batch", 16) // 2,
                iou=self.config.get("iou_threshold", 0.5),
                conf=self.config.get("conf_threshold", 0.25),
                project=str(eval_output_dir),
                name=run_name,
                save_json=True
            )
            parsed_results = self._parse_and_log_metrics(metrics)
            
        except Exception as e:
            logger.error(f"❌ An error occurred during model evaluation: {e}", exc_info=True)
            return {}
        
        logger.info("✅ Final evaluation complete.")
        return parsed_results

    def _parse_and_log_metrics(self, metrics: DetMetrics) -> Dict[str, Any]:
        """
        Parse `DetMetrics` to compute base vs. novel metrics and log summaries.
        """
        logger.info("Parsing per-class metrics to calculate base vs. novel performance.")
        results = {}
        
        results["overall"] = {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map
        }
        
        base_ap50_scores, novel_ap50_scores = [], []
        base_map_scores, novel_map_scores = [], []

        map_per_class = metrics.box.maps
        ap50_per_class = metrics.box.ap50

        for i in range(self.dm.num_total_classes):
            class_indices_in_results = np.where(metrics.box.ap_class_index == i)[0]
            
            if class_indices_in_results.size > 0:
                result_idx = class_indices_in_results[0]
                ap50 = ap50_per_class[result_idx]
                map_val = map_per_class[result_idx]
            else:
                ap50, map_val = 0.0, 0.0

            if i in self.base_class_ids:
                base_ap50_scores.append(ap50)
                base_map_scores.append(map_val)
            elif i in self.novel_class_ids:
                novel_ap50_scores.append(ap50)
                novel_map_scores.append(map_val)

        results["base_classes"] = {
            "mAP50": np.mean(base_ap50_scores) if base_ap50_scores else 0.0,
            "mAP50-95": np.mean(base_map_scores) if base_map_scores else 0.0
        }
        results["novel_classes"] = {
            "mAP50": np.mean(novel_ap50_scores) if novel_ap50_scores else 0.0,
            "mAP50-95": np.mean(novel_map_scores) if novel_map_scores else 0.0
        }

        # Create and print the summary table with decimal points
        summary_data_decimal = {
            "Metric": ["mAP@50", "mAP@50-95"],
            "Base Classes": [f"{results['base_classes']['mAP50']:.4f}", f"{results['base_classes']['mAP50-95']:.4f}"],
            "Novel Classes": [f"{results['novel_classes']['mAP50']:.4f}", f"{results['novel_classes']['mAP50-95']:.4f}"],
            "Overall": [f"{results['overall']['mAP50']:.4f}", f"{results['overall']['mAP50-95']:.4f}"]
        }
        summary_df_decimal = pd.DataFrame(summary_data_decimal)
        logger.info("\n" + "="*60 + "\n📊 Final Performance Summary (Decimal)\n" + "="*60 + "\n" + summary_df_decimal.to_string(index=False))
        
        # Create and print the summary table with percentages
        summary_data_percent = {
            "Metric": ["mAP@50", "mAP@50-95"],
            "Base Classes": [f"{results['base_classes']['mAP50']:.2%}", f"{results['base_classes']['mAP50-95']:.2%}"],
            "Novel Classes": [f"{results['novel_classes']['mAP50']:.2%}", f"{results['novel_classes']['mAP50-95']:.2%}"],
            "Overall": [f"{results['overall']['mAP50']:.2%}", f"{results['overall']['mAP50-95']:.2%}"]
        }
        summary_df_percent = pd.DataFrame(summary_data_percent)
        logger.info("\n" + "="*60 + "\n📊 Final Performance Summary (Percentage)\n" + "="*60 + "\n" + summary_df_percent.to_string(index=False))
        
        return results