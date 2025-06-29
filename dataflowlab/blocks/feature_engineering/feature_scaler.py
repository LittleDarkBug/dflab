from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class FeatureScaler(BlockBase):
    """
    Bloc de mise à l'échelle des variables numériques (StandardScaler, MinMaxScaler, etc.).
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "FeatureScaler", params=params, category="feature_engineering")
        self.scaler = None
        self.numeric_columns = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'FeatureScaler':
        """Ajuste le scaler sur les données."""
        try:
            if X is None or X.empty:
                raise ValueError("Aucune donnée pour l'ajustement")
            
            # Identification des colonnes numériques
            self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if not self.numeric_columns:
                logger.warning("Aucune colonne numérique trouvée pour la mise à l'échelle")
                self._fitted = True
                return self
            
            # Configuration du scaler
            scaler_type = self.params.get("scaler_type", "standard")
            
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                feature_range = self.params.get("feature_range", (0, 1))
                self.scaler = MinMaxScaler(feature_range=feature_range)
            elif scaler_type == "robust":
                self.scaler = RobustScaler()
            elif scaler_type == "maxabs":
                self.scaler = MaxAbsScaler()
            else:
                raise ValueError(f"Type de scaler non supporté: {scaler_type}")
            
            # Ajustement du scaler
            self.scaler.fit(X[self.numeric_columns])
            
            self._fitted = True
            logger.info(f"FeatureScaler ajusté ({scaler_type}) sur {len(self.numeric_columns)} colonnes")
            
            return self
            
        except Exception as e:
            error_msg = f"Erreur lors de l'ajustement du scaler: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Applique la mise à l'échelle aux colonnes numériques.
        
        Args:
            data: DataFrame à transformer
            
        Returns:
            DataFrame avec variables mises à l'échelle
        """
        try:
            if data is None or data.empty:
                raise ValueError("Aucune donnée à traiter")
            
            if not self._fitted:
                self.fit(data)
            
            if not self.numeric_columns or self.scaler is None:
                logger.warning("Pas de transformation à appliquer")
                return data.copy()
            
            result = data.copy()
            
            # Application de la transformation
            scaled_data = self.scaler.transform(data[self.numeric_columns])
            result[self.numeric_columns] = scaled_data
            
            self._output_data = result
            logger.info(f"Mise à l'échelle appliquée à {len(self.numeric_columns)} colonnes")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur lors de la mise à l'échelle: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "scaler_type",
                "type": "select",
                "label": "Type de scaler",
                "default": "standard",
                "options": ["standard", "minmax", "robust", "maxabs"],
                "required": True
            },
            {
                "name": "feature_range_min",
                "type": "number",
                "label": "Min range (MinMaxScaler)",
                "default": 0,
                "step": 0.1
            },
            {
                "name": "feature_range_max",
                "type": "number",
                "label": "Max range (MinMaxScaler)",
                "default": 1,
                "step": 0.1
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('FeatureScaler', FeatureScaler)
