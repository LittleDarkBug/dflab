from typing import Any, Dict, Optional
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class AnomalyDetectionBlock(BlockBase):
    """
    Bloc de détection d'anomalies (Isolation Forest, One-Class SVM).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="AnomalyDetection", params=params, category="unsupervised")
        self.logger = get_logger("AnomalyDetection")
        self.model = None

    def fit(self, X, y=None):
        algo = self.params.get("algo", "isolation_forest")
        if algo == "oneclass_svm":
            self.model = OneClassSVM()
        else:
            self.model = IsolationForest()
        self.model.fit(X)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.predict(X), index=X.index, name="anomaly")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute anomaly detection."""
        try:
            # Sélection des colonnes numériques
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                self.logger.error("Aucune colonne numérique trouvée pour la détection d'anomalies")
                return data
            
            X = data[numeric_cols].fillna(0)  # Simple imputation
            
            # Fit et prédiction
            self.fit(X)
            anomalies = self.transform(X)
            
            # Ajouter les résultats
            result_data = data.copy()
            result_data['anomaly'] = anomalies
            
            self.logger.info(f"Détection d'anomalies terminée")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'anomalies : {str(e)}")
            return data

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Méthode process pour compatibilité avec BlockBase.
        """
        return self.execute(data, **kwargs)

# Auto-enregistrement du bloc
BlockRegistry.register_block('AnomalyDetectionBlock', AnomalyDetectionBlock)
