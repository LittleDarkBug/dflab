from typing import Any, Dict, Optional
import pandas as pd
from sklearn.cluster import DBSCAN
from dataflowlab.core.block_base import BlockBase
from dataflowlab.utils.logger import get_logger
# Auto-enregistrement du bloc
from dataflowlab.core.block_registry import BlockRegistry

class DBSCANBlock(BlockBase):
    """
    Bloc de clustering basé densité (DBSCAN).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="DBSCAN", params=params, category="unsupervised")
        self.logger = get_logger("DBSCAN")
        self.model = None

    def fit(self, X, y=None):
        eps = self.params.get("eps", 0.5)
        min_samples = self.params.get("min_samples", 5)
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.model.fit(X)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("DBSCAN non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.fit_predict(X), index=X.index, name="cluster")

# Enregistrement du bloc
BlockRegistry.register_block('DBSCANBlock', DBSCANBlock)
