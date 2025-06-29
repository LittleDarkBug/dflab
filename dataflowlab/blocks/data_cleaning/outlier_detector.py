from typing import Any, Dict, Optional
import pandas as pd
from sklearn.ensemble import IsolationForest
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class OutlierDetector(BlockBase):
    """
    Bloc de détection d'outliers (IQR, Z-score, Isolation Forest).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="OutlierDetector", params=params, category="data_cleaning")
        self.logger = get_logger("OutlierDetector")
        self.method = self.params.get("method", "IQR")
        self.threshold = self.params.get("threshold", 1.5)
        self.model = None

    def fit(self, X, y=None):
        if self.method == "isolation_forest":
            self.model = IsolationForest()
            self.model.fit(X)
        return self

    def transform(self, X):
        method = self.method.lower()
        
        if method == "iqr":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            threshold = self.threshold
            mask = ~((X < (Q1 - threshold * IQR)) | (X > (Q3 + threshold * IQR))).any(axis=1)
            return X[mask]
        elif method == "zscore" or method == "z-score":
            z = (X - X.mean()) / X.std()
            threshold = self.threshold if self.threshold > 1 else 3  # Z-score threshold should be > 1
            mask = (z.abs() < threshold).all(axis=1)
            return X[mask]
        elif method == "isolation_forest":
            if self.model is None:
                raise RuntimeError("IsolationForest non entraîné.")
            mask = self.model.predict(X) == 1
            return X[mask]
        else:
            raise ValueError(f"Méthode inconnue: {method}. Méthodes supportées: iqr, zscore, isolation_forest")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data by detecting and removing outliers."""
        try:
            self.logger.info(f"Détection d'outliers avec méthode: {self.method}")
            
            # Sélectionner seulement les colonnes numériques
            numeric_cols = data.select_dtypes(include=[int, float]).columns
            if len(numeric_cols) == 0:
                self.logger.warning("Aucune colonne numérique trouvée pour la détection d'outliers")
                return data
            
            numeric_data = data[numeric_cols]
            
            # Appliquer la détection d'outliers
            if self.method == "isolation_forest":
                # Pour isolation forest, d'abord fit puis transform
                self.fit(numeric_data)
                
            cleaned_data = self.transform(numeric_data)
            
            # Recombiner avec les colonnes non-numériques
            if len(numeric_cols) < len(data.columns):
                non_numeric_cols = data.columns.difference(numeric_cols)
                # Utiliser l'index pour recombiner correctement
                result = pd.concat([
                    cleaned_data,
                    data.loc[cleaned_data.index, non_numeric_cols]
                ], axis=1)
                # Réordonner les colonnes comme dans l'original
                result = result[data.columns]
            else:
                result = cleaned_data
            
            self.logger.info(f"Outliers détectés: {len(data)} → {len(result)} lignes restantes")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'outliers: {str(e)}")
            # En cas d'erreur, retourner les données originales
            return data

# Auto-enregistrement du bloc
BlockRegistry.register_block('OutlierDetector', OutlierDetector)
