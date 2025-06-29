from typing import Any, Dict, Optional
import pandas as pd
import re
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class TextPreprocessor(BlockBase):
    """
    Bloc de nettoyage et tokenisation de texte.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="TextPreprocessor", params=params, category="advanced")
        self.logger = get_logger("TextPreprocessor")

    def transform(self, X):
        col = self.params.get("col")
        X_new = X.copy()
        X_new[col] = X_new[col].astype(str).str.lower()
        X_new[col] = X_new[col].apply(lambda t: re.sub(r"[^\w\s]", "", t))
        X_new[col] = X_new[col].str.split()
        return X_new

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute text preprocessing."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            text_column = self.params.get('text_column', 'text')
            
            if text_column not in data.columns:
                self.logger.error(f"Colonne texte '{text_column}' non trouvée")
                return data
            
            result_data = data.copy()
            
            # Prétraitement simple
            result_data[f'{text_column}_processed'] = result_data[text_column].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
            
            self.logger.info(f"Prétraitement de texte terminé pour la colonne '{text_column}'")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement de texte : {str(e)}")
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
BlockRegistry.register_block('TextPreprocessor', TextPreprocessor)
