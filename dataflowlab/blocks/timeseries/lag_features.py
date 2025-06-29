from typing import Any, Dict, Optional
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class LagFeatures(BlockBase):
    """
    Bloc de création de variables retardées (lag features).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="LagFeatures", params=params, category="timeseries")
        self.logger = get_logger("LagFeatures")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute lag features creation."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des lag features : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process method for compatibility."""
        return self.execute(data, **kwargs)

    def transform(self, X):
        lags = self.params.get("lags", [1])
        col = self.params.get("col")
        X_new = X.copy()
        for lag in lags:
            X_new[f"{col}_lag{lag}"] = X[col].shift(lag)
        return X_new

# Auto-enregistrement du bloc
BlockRegistry.register_block('LagFeatures', LagFeatures)
