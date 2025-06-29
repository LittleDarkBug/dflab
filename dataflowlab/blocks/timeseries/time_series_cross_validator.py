from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class TimeSeriesCrossValidator(BlockBase):
    """
    Bloc de validation croisée temporelle.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="TimeSeriesCrossValidator", params=params, category="timeseries")
        self.logger = get_logger("TimeSeriesCrossValidator")

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute time series cross validation."""
        try:
            splits = self.transform(data)
            return {"splits": splits}
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation croisée temporelle : {str(e)}")
            return {"error": str(e)}

    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Process method for compatibility."""
        return self.execute(data, **kwargs)

    def transform(self, X):
        n_splits = self.params.get("n_splits", 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
        return splits

# Auto-enregistrement du bloc
BlockRegistry.register_block('TimeSeriesCrossValidator', TimeSeriesCrossValidator)
