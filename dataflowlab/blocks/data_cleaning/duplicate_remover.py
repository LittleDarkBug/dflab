"""
Bloc de suppression des doublons.
"""

from typing import Any, Dict, Optional
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class DuplicateRemoverBlock(BlockBase):
    """
    Bloc de suppression des lignes dupliquées.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'subset': None,  # Colonnes à considérer (None = toutes)
            'keep': 'first',  # 'first', 'last', False
            'ignore_index': False
        }
        super().__init__(
            name="DuplicateRemover",
            category="data_cleaning",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
    
    def transform(self, X):
        """Méthode transform pour compatibilité."""
        subset = self.params.get("subset", None)
        keep = self.params.get("keep", "first")
        return X.drop_duplicates(subset=subset, keep=keep)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute duplicate removal."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            initial_count = len(data)
            subset = self.params.get('subset')
            keep = self.params.get('keep', 'first')
            ignore_index = self.params.get('ignore_index', False)
            
            # Si subset est spécifié, convertir en liste
            if subset and isinstance(subset, str):
                subset = [col.strip() for col in subset.split(',')]
                # Vérifier que les colonnes existent
                missing_cols = set(subset) - set(data.columns)
                if missing_cols:
                    self.logger.warning(f"Colonnes introuvables: {missing_cols}")
                    subset = [col for col in subset if col in data.columns]
                    if not subset:
                        subset = None
            
            # Supprimer les doublons
            result_data = data.drop_duplicates(
                subset=subset,
                keep=keep,
                ignore_index=ignore_index
            )
            
            removed_count = initial_count - len(result_data)
            
            self.logger.info(f"Doublons supprimés: {removed_count} lignes sur {initial_count}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression des doublons : {str(e)}")
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
BlockRegistry.register_block('DuplicateRemoverBlock', DuplicateRemoverBlock)
