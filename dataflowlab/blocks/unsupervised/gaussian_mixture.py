from typing import Any, Dict, Optional
import pandas as pd
from sklearn.mixture import GaussianMixture
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class GaussianMixtureBlock(BlockBase):
    """
    Bloc de modèles de mélange gaussien (Gaussian Mixture).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="GaussianMixture", params=params, category="unsupervised")
        self.logger = get_logger("GaussianMixture")
        self.model = None

    def fit(self, X, y=None):
        n_components = self.params.get("n_components", 2)
        self.model = GaussianMixture(n_components=n_components)
        self.model.fit(X)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("GaussianMixture non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.predict(X), index=X.index, name="cluster")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute Gaussian Mixture clustering."""
        try:
            self.fit(data)
            clusters = self.transform(data)
            result = data.copy()
            result['cluster'] = clusters
            return result
        except Exception as e:
            self.logger.error(f"Erreur lors du clustering Gaussian Mixture : {str(e)}")
            return data

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Méthode process pour compatibilité avec BlockBase.
        """
        return self.execute(data, **kwargs)

# Auto-enregistrement du bloc
BlockRegistry.register_block('GaussianMixture', GaussianMixtureBlock)
