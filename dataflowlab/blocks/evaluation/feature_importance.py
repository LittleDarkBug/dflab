from typing import Any, Dict, Optional
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.utils.logger import get_logger

class FeatureImportanceBlock(BlockBase):
    """
    Bloc d'importance des variables (feature importance).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="FeatureImportance", params=params)
        self.logger = get_logger("FeatureImportance")

    def transform(self, X):
        model = self.params.get("model")
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            return model.coef_
        else:
            raise ValueError("Le modèle ne fournit pas d'importance de features.")
