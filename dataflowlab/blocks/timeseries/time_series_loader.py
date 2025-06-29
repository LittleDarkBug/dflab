from typing import Any, Dict, Optional
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class TimeSeriesLoader(BlockBase):
    """
    Bloc de chargement de données temporelles.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="TimeSeriesLoader", params=params, category="timeseries")
        self.logger = get_logger("TimeSeriesLoader")

    def execute(self, data: Any = None, **kwargs) -> pd.DataFrame:
        """Execute time series loading."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des séries temporelles : {str(e)}")
            return pd.DataFrame()

    def process(self, data: Any = None, **kwargs) -> pd.DataFrame:
        """Process method for compatibility."""
        return self.execute(data, **kwargs)

    def transform(self, X=None):
        try:
            path = self.params.get("path")
            date_col = self.params.get("date_col")
            df = pd.read_csv(path, parse_dates=[date_col])
            df = df.sort_values(date_col)
            return df
        except Exception as e:
            self.logger.error(f"Erreur chargement TimeSeries: {e}")
            raise

# Auto-enregistrement du bloc
BlockRegistry.register_block('TimeSeriesLoader', TimeSeriesLoader)
