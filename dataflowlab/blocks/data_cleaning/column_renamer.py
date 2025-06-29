"""
Bloc de renommage de colonnes.
"""

from typing import Any, Dict, Optional
import pandas as pd
import re
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class ColumnRenamerBlock(BlockBase):
    """
    Bloc de renommage de colonnes avec diverses options.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'mapping': {},  # Mapping manuel {ancien_nom: nouveau_nom}
            'auto_clean': True,  # Nettoyage automatique des noms
            'to_lowercase': True,  # Conversion en minuscules
            'replace_spaces': True,  # Remplacer espaces par underscore
            'remove_special_chars': True,  # Supprimer caractères spéciaux
            'prefix': '',  # Préfixe à ajouter
            'suffix': ''   # Suffixe à ajouter
        }
        super().__init__(
            name="ColumnRenamer",
            category="data_cleaning",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
    
    def transform(self, X):
        """Méthode transform pour compatibilité."""
        mapping = self.params.get("mapping", None)
        if mapping:
            return X.rename(columns=mapping)
        else:
            # Nettoyage basique : strip et lower
            return X.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute column renaming."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            result_data = data.copy()
            rename_map = {}
            
            # Mapping manuel
            manual_mapping = self.params.get('mapping', {})
            if manual_mapping:
                rename_map.update(manual_mapping)
            
            # Nettoyage automatique
            if self.params.get('auto_clean', True):
                for col in result_data.columns:
                    if col not in rename_map:  # Ne pas écraser le mapping manuel
                        new_name = self._clean_column_name(col)
                        if new_name != col:
                            rename_map[col] = new_name
            
            # Appliquer les renommages
            if rename_map:
                result_data = result_data.rename(columns=rename_map)
                changes = [f"{old} -> {new}" for old, new in rename_map.items()]
                self.logger.info(f"Colonnes renommées: {', '.join(changes)}")
            else:
                self.logger.info("Aucun renommage nécessaire")
                
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du renommage des colonnes : {str(e)}")
            return data
    
    def _clean_column_name(self, col_name: str) -> str:
        """Nettoyage automatique d'un nom de colonne."""
        new_name = str(col_name)
        
        # Conversion en minuscules
        if self.params.get('to_lowercase', True):
            new_name = new_name.lower()
        
        # Remplacer les espaces
        if self.params.get('replace_spaces', True):
            new_name = re.sub(r'\s+', '_', new_name)
        
        # Supprimer les caractères spéciaux
        if self.params.get('remove_special_chars', True):
            new_name = re.sub(r'[^a-zA-Z0-9_]', '', new_name)
        
        # Supprimer les underscores multiples
        new_name = re.sub(r'_+', '_', new_name)
        
        # Supprimer les underscores en début/fin
        new_name = new_name.strip('_')
        
        # S'assurer que le nom commence par une lettre ou underscore
        if new_name and not re.match(r'^[a-zA-Z_]', new_name):
            new_name = 'col_' + new_name
        
        # Préfixe et suffixe
        prefix = self.params.get('prefix', '')
        suffix = self.params.get('suffix', '')
        
        if prefix:
            new_name = prefix + new_name
        if suffix:
            new_name = new_name + suffix
        
        # Si le nom est vide après nettoyage
        if not new_name:
            new_name = 'unnamed_column'
        
        return new_name



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
BlockRegistry.register_block('ColumnRenamerBlock', ColumnRenamerBlock)
