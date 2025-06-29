"""
Bloc de filtrage de lignes.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class RowFilterBlock(BlockBase):
    """
    Bloc de filtrage de lignes selon diverses conditions.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'filter_type': 'simple',  # 'simple', 'query', 'custom'
            'column': None,  # Colonne pour le filtrage simple
            'operator': '==',  # ==, !=, >, <, >=, <=, in, not_in
            'value': None,  # Valeur de référence
            'query_string': '',  # String de requête pandas
            'remove_nulls': False,  # Supprimer les lignes avec des nulls
            'null_columns': [],  # Colonnes spécifiques pour les nulls
        }
        super().__init__(
            name="RowFilter",
            category="data_cleaning",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
    
    def transform(self, X):
        """Méthode transform pour compatibilité."""
        condition = self.params.get("condition")
        if condition is None:
            return X
        return X.query(condition)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute row filtering."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            result_data = data.copy()
            initial_count = len(result_data)
            
            # Filtrage des nulls
            if self.params.get('remove_nulls', False):
                result_data = self._filter_nulls(result_data)
            
            # Filtrage selon le type
            filter_type = self.params.get('filter_type', 'simple')
            
            if filter_type == 'simple':
                result_data = self._simple_filter(result_data)
            elif filter_type == 'query':
                result_data = self._query_filter(result_data)
            
            final_count = len(result_data)
            removed_count = initial_count - final_count
            
            self.logger.info(f"Filtrage terminé: {removed_count} lignes supprimées sur {initial_count}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du filtrage des lignes : {str(e)}")
            return data
    
    def _filter_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtrage des valeurs nulles."""
        null_columns = self.params.get('null_columns', [])
        
        if null_columns:
            # Supprimer les lignes avec des nulls dans les colonnes spécifiées
            valid_columns = [col for col in null_columns if col in data.columns]
            if valid_columns:
                result = data.dropna(subset=valid_columns)
                self.logger.info(f"Lignes avec nulls supprimées dans {valid_columns}")
                return result
        else:
            # Supprimer les lignes avec des nulls dans toutes les colonnes
            result = data.dropna()
            self.logger.info("Lignes avec nulls supprimées (toutes colonnes)")
            return result
            
        return data
    
    def _simple_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtrage simple sur une colonne."""
        column = self.params.get('column')
        operator = self.params.get('operator', '==')
        value = self.params.get('value')
        
        if not column or column not in data.columns:
            self.logger.warning(f"Colonne '{column}' introuvable pour le filtrage")
            return data
            
        if value is None:
            self.logger.warning("Aucune valeur spécifiée pour le filtrage")
            return data
        
        col_data = data[column]
        
        try:
            if operator == '==':
                mask = col_data == value
            elif operator == '!=':
                mask = col_data != value
            elif operator == '>':
                mask = col_data > value
            elif operator == '<':
                mask = col_data < value
            elif operator == '>=':
                mask = col_data >= value
            elif operator == '<=':
                mask = col_data <= value
            elif operator == 'in':
                # value doit être une liste
                if isinstance(value, str):
                    value = [v.strip() for v in value.split(',')]
                mask = col_data.isin(value)
            elif operator == 'not_in':
                # value doit être une liste
                if isinstance(value, str):
                    value = [v.strip() for v in value.split(',')]
                mask = ~col_data.isin(value)
            elif operator == 'contains':
                mask = col_data.astype(str).str.contains(str(value), na=False)
            elif operator == 'not_contains':
                mask = ~col_data.astype(str).str.contains(str(value), na=False)
            else:
                self.logger.warning(f"Opérateur '{operator}' non reconnu")
                return data
            
            result = data[mask]
            self.logger.info(f"Filtrage simple: {column} {operator} {value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du filtrage simple : {str(e)}")
            return data
    
    def _query_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtrage avec une requête pandas."""
        query_string = self.params.get('query_string', '')
        
        if not query_string:
            self.logger.warning("Aucune requête spécifiée")
            return data
        
        try:
            result = data.query(query_string)
            self.logger.info(f"Filtrage par requête: {query_string}")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du filtrage par requête : {str(e)}")
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
BlockRegistry.register_block('RowFilterBlock', RowFilterBlock)
