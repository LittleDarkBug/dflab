from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import learning_curve
from dataflowlab.core.block_base import BlockBase
from dataflowlab.utils.logger import get_logger

class LearningCurvesBlock(BlockBase):
    """
    Bloc de génération de courbes d'apprentissage.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="LearningCurves", params=params)
        self.logger = get_logger("LearningCurves")

    def transform(self, X):
        estimator = self.params.get("estimator")
        X_data = self.params.get("X")
        y_data = self.params.get("y")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_data, y_data)
        return {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "test_scores": test_scores
        }
