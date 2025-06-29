from typing import Any, Dict, Optional
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class DateFeatureExtractor(BlockBase):
    """
    Bloc d'extraction de composants temporels à partir de dates.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="DateFeatureExtractor", params=params, category="feature_engineering")
        self.logger = get_logger("DateFeatureExtractor")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extrait les composants temporels des colonnes de dates.
        """
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de features temporelles : {str(e)}")
            return data

    def transform(self, X):
        X_new = X.copy()
        for col in X.columns:
            X_new[f"{col}_year"] = pd.to_datetime(X[col]).dt.year
            X_new[f"{col}_month"] = pd.to_datetime(X[col]).dt.month
            X_new[f"{col}_day"] = pd.to_datetime(X[col]).dt.day
            X_new[f"{col}_weekday"] = pd.to_datetime(X[col]).dt.weekday
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
BlockRegistry.register_block('DateFeatureExtractor', DateFeatureExtractor)
