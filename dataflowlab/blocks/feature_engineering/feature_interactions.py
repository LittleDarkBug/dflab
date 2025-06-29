from typing import Any, Dict, Optional
import pandas as pd
from itertools import combinations
from dataflowlab.core.block_base import BlockBase
from dataflowlab.utils.logger import get_logger
# Auto-enregistrement du bloc
from dataflowlab.core.block_registry import BlockRegistry

class FeatureInteractions(BlockBase):
    """
    Bloc de création d'interactions entre variables.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="FeatureInteractions", params=params, category="feature_engineering")
        self.logger = get_logger("FeatureInteractions")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Crée des interactions entre variables numériques.
        """
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de la création d'interactions : {str(e)}")
            return data

    def transform(self, X):
        cols = X.columns
        X_new = X.copy()
        for c1, c2 in combinations(cols, 2):
            X_new[f"{c1}_x_{c2}"] = X[c1] * X[c2]
        return X_new


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
BlockRegistry.register_block('FeatureInteractions', FeatureInteractions)
