from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class LabelEncoder(BlockBase):
    """
    Bloc d'encodage ordinal des variables catégorielles.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="LabelEncoder", category="feature_engineering", params=params)
        self.logger = get_logger("LabelEncoder")
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = SkLabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_enc = X.copy()
        for col, le in self.encoders.items():
            X_enc[col] = le.transform(X[col].astype(str))
        return X_enc
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute label encoding."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            categorical_columns = self.params.get('categorical_columns', [])
            if not categorical_columns:
                # Auto-détection des colonnes catégorielles
                categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            result_data = data.copy()
            
            for col in categorical_columns:
                if col in data.columns:
                    le = SkLabelEncoder()
                    # Traiter les valeurs manquantes
                    mask = result_data[col].notna()
                    if mask.any():
                        result_data.loc[mask, f'{col}_encoded'] = le.fit_transform(result_data.loc[mask, col].astype(str))
                        self.encoders[col] = le
                        self.logger.info(f"Colonne '{col}' encodée")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'encodage : {str(e)}")
            return data



    def process(self, data):
        """Process data using this block."""
        try:
            self.logger.info(f"Traitement des données avec {self.__class__.__name__}")
            
            # Si le bloc a une méthode transform, l'utiliser
            if hasattr(self, 'transform') and callable(getattr(self, 'transform')):
                return self.transform(data)
            
            # Si le bloc a une méthode fit_transform, l'utiliser
            elif hasattr(self, 'fit_transform') and callable(getattr(self, 'fit_transform')):
                return self.fit_transform(data)
            
            # Si le bloc a fit et transform séparément
            elif hasattr(self, 'fit') and hasattr(self, 'transform'):
                self.fit(data)
                return self.transform(data)
            
            # Sinon, retourner les données telles quelles
            else:
                self.logger.warning(f"{self.__class__.__name__} n'a pas de méthode de traitement définie")
                return data
                
        except Exception as e:
            self.logger.error(f"Erreur dans {self.__class__.__name__}: {str(e)}")
            # En cas d'erreur, retourner les données originales
            return data


# Auto-enregistrement du bloc
BlockRegistry.register_block('LabelEncoder', LabelEncoder)
