from typing import Any, Dict, Optional
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class RollingStatistics(BlockBase):
    """
    Bloc de calcul de moyennes/écarts-types mobiles.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="RollingStatistics", params=params, category="timeseries")
        self.logger = get_logger("RollingStatistics")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute rolling statistics calculation."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des statistiques mobiles : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process method for compatibility."""
        return self.execute(data, **kwargs)

    def transform(self, X):
        window = self.params.get("window", 3)
        col = self.params.get("col")
        X_new = X.copy()
        X_new[f"{col}_rollmean"] = X[col].rolling(window).mean()
        X_new[f"{col}_rollstd"] = X[col].rolling(window).std()
        return X_new

# Auto-enregistrement du bloc
BlockRegistry.register_block('RollingStatistics', RollingStatistics)
