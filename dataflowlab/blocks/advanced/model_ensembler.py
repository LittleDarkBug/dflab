from typing import Any, Dict, Optional, List
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class ModelEnsembler(BlockBase):
    """
    Bloc d'ensemble de modèles (voting, stacking).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="ModelEnsembler", params=params, category="advanced")
        self.logger = get_logger("ModelEnsembler")
        self.ensemble = None

    def fit(self, X, y):
        method = self.params.get("method", "voting")
        estimators: List = self.params.get("estimators", [])
        if method == "stacking":
            self.ensemble = StackingClassifier(estimators=estimators)
        else:
            self.ensemble = VotingClassifier(estimators=estimators, voting="hard")
        self.ensemble.fit(X, y)
        return self

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute model ensembling."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ensemble de modèles : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Méthode process pour compatibilité avec BlockBase."""
        return self.execute(data, **kwargs)

    def transform(self, X):
        if self.ensemble is None:
            raise RuntimeError("Ensemble non entraîné. Appelez fit d'abord.")
        return pd.Series(self.ensemble.predict(X), index=X.index, name="prediction")

# Auto-enregistrement du bloc
BlockRegistry.register_block('ModelEnsembler', ModelEnsembler)
