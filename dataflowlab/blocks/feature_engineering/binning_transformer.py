from typing import Any, Dict, Optional
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from dataflowlab.core.block_base import BlockBase
from dataflowlab.utils.logger import get_logger
# Auto-enregistrement du bloc
from dataflowlab.core.block_registry import BlockRegistry

class BinningTransformer(BlockBase):
    """
    Bloc de discrétisation des variables continues.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="BinningTransformer", params=params)
        self.logger = get_logger("BinningTransformer")
        self.binner = None

    def fit(self, X, y=None):
        n_bins = self.params.get("n_bins", 5)
        strategy = self.params.get("strategy", "uniform")
        self.binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
        self.binner.fit(X)
        return self

    def transform(self, X):
        if self.binner is None:
            raise RuntimeError("Binner non initialisé. Appelez fit d'abord.")
        arr = self.binner.transform(X)
        columns = [f"bin_{col}" for col in X.columns]
        return pd.DataFrame(arr, columns=columns, index=X.index)
    
    def process(self, data):
        """Méthode pour la compatibilité avec le pipeline."""
        if data is None or data.empty:
            raise ValueError("Aucune donnée fournie pour le traitement")
        
        # Si c'est un tuple (X, y), on traite seulement X
        if isinstance(data, tuple):
            X, y = data
            X_transformed = self.fit(X).transform(X)
            return X_transformed, y
        else:
            # Si c'est juste X
            return self.fit(data).transform(data)

# Enregistrement du bloc
BlockRegistry.register_block('BinningTransformer', BinningTransformer)
