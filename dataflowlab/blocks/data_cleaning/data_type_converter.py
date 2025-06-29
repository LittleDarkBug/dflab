"""
Bloc de conversion de types de données.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class DataTypeConverterBlock(BlockBase):
    """
    Bloc de conversion automatique ou manuelle des types de données.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'auto_convert': True,  # Conversion automatique
            'manual_conversions': {},  # Conversions manuelles {colonne: type}
            'numeric_errors': 'coerce',  # 'raise', 'coerce', 'ignore'
            'datetime_format': None  # Format datetime
        }
        super().__init__(
            name="DataTypeConverter",
            category="data_cleaning",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
    
    def transform(self, X):
        """Méthode transform pour compatibilité."""
        return X.convert_dtypes()
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute data type conversion."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            result_data = data.copy()
            conversions_made = []
            
            # Conversion automatique
            if self.params.get('auto_convert', True):
                result_data = self._auto_convert_types(result_data, conversions_made)
            
            # Conversions manuelles
            manual_conversions = self.params.get('manual_conversions', {})
            if manual_conversions:
                result_data = self._manual_convert_types(result_data, manual_conversions, conversions_made)
            
            if conversions_made:
                self.logger.info(f"Conversions effectuées: {', '.join(conversions_made)}")
            else:
                self.logger.info("Aucune conversion nécessaire")
                
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion des types : {str(e)}")
            return data
    
    def _auto_convert_types(self, data: pd.DataFrame, conversions_made: list) -> pd.DataFrame:
        """Conversion automatique des types."""
        result = data.copy()
        
        for col in result.columns:
            if result[col].dtype == 'object':
                # Essayer de convertir en numérique
                numeric_result = pd.to_numeric(result[col], errors='coerce')
                if not numeric_result.isna().all():
                    # Si au moins une valeur peut être convertie
                    non_null_ratio = numeric_result.notna().sum() / len(numeric_result)
                    if non_null_ratio > 0.8:  # Si au moins 80% des valeurs sont convertibles
                        result[col] = numeric_result
                        conversions_made.append(f"{col}: object -> numeric")
                        continue
                
                # Essayer de convertir en datetime
                try:
                    datetime_format = self.params.get('datetime_format')
                    if datetime_format:
                        datetime_result = pd.to_datetime(result[col], format=datetime_format, errors='coerce')
                    else:
                        datetime_result = pd.to_datetime(result[col], errors='coerce', infer_datetime_format=True)
                    
                    if not datetime_result.isna().all():
                        non_null_ratio = datetime_result.notna().sum() / len(datetime_result)
                        if non_null_ratio > 0.8:
                            result[col] = datetime_result
                            conversions_made.append(f"{col}: object -> datetime")
                            continue
                except:
                    pass
                
                # Essayer de convertir en catégorie si peu de valeurs uniques
                unique_ratio = result[col].nunique() / len(result[col])
                if unique_ratio < 0.1:  # Moins de 10% de valeurs uniques
                    result[col] = result[col].astype('category')
                    conversions_made.append(f"{col}: object -> category")
        
        return result
    
    def _manual_convert_types(self, data: pd.DataFrame, conversions: dict, conversions_made: list) -> pd.DataFrame:
        """Conversions manuelles spécifiées."""
        result = data.copy()
        numeric_errors = self.params.get('numeric_errors', 'coerce')
        
        for col, target_type in conversions.items():
            if col not in result.columns:
                self.logger.warning(f"Colonne '{col}' introuvable, conversion ignorée")
                continue
                
            try:
                if target_type == 'numeric':
                    result[col] = pd.to_numeric(result[col], errors=numeric_errors)
                elif target_type == 'datetime':
                    datetime_format = self.params.get('datetime_format')
                    if datetime_format:
                        result[col] = pd.to_datetime(result[col], format=datetime_format, errors='coerce')
                    else:
                        result[col] = pd.to_datetime(result[col], errors='coerce')
                elif target_type == 'category':
                    result[col] = result[col].astype('category')
                elif target_type == 'string':
                    result[col] = result[col].astype('string')
                elif target_type in ['int64', 'float64', 'bool']:
                    result[col] = result[col].astype(target_type)
                else:
                    self.logger.warning(f"Type '{target_type}' non reconnu pour la colonne '{col}'")
                    continue
                    
                conversions_made.append(f"{col}: -> {target_type}")
                
            except Exception as e:
                self.logger.error(f"Erreur conversion {col} -> {target_type}: {str(e)}")
        
        return result



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
BlockRegistry.register_block('DataTypeConverterBlock', DataTypeConverterBlock)
