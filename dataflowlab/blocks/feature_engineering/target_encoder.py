"""
Bloc Target Encoder pour l'encodage basé sur la cible.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class TargetEncoderBlock(BlockBase):
    """
    Bloc d'encodage basé sur la variable cible (Target Encoding).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'target_column': None,
            'categorical_columns': [],
            'smoothing': 1.0
        }
        super().__init__(
            name="TargetEncoder",
            category="feature_engineering",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.mapping = {}

    def fit(self, X, y):
        """Méthode fit pour compatibilité."""
        for col in X.columns:
            means = y.groupby(X[col]).mean()
            self.mapping[col] = means
        return self

    def transform(self, X):
        """Méthode transform pour compatibilité."""
        X_enc = X.copy()
        for col, mapping in self.mapping.items():
            X_enc[col] = X[col].map(mapping)
        return X_enc
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute target encoding."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            target_column = self.params.get('target_column')
            categorical_columns = self.params.get('categorical_columns', [])
            
            if not target_column or target_column not in data.columns:
                self.logger.error(f"Colonne cible '{target_column}' non trouvée")
                return data
            
            if not categorical_columns:
                # Auto-détection des colonnes catégorielles
                categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
                if target_column in categorical_columns:
                    categorical_columns.remove(target_column)
            
            result_data = data.copy()
            target_data = result_data[target_column]
            
            for col in categorical_columns:
                if col in result_data.columns:
                    means = target_data.groupby(result_data[col]).mean()
                    result_data[f'{col}_target_encoded'] = result_data[col].map(means)
            
            self.logger.info(f"Target encoding appliqué à {len(categorical_columns)} colonnes")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du target encoding : {str(e)}")
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
BlockRegistry.register_block('TargetEncoderBlock', TargetEncoderBlock)
