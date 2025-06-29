"""
Bloc Time Series Decomposition.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class TimeSeriesDecompositionBlock(BlockBase):
    """
    Bloc de décomposition de séries temporelles.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            name="TimeSeriesDecomposition",
            category="timeseries",
            params=params
        )
        self.logger = get_logger(self.__class__.__name__)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute time series decomposition."""
        if data is None or data.empty:
            return pd.DataFrame()
            
        try:
            value_column = self.params.get('value_column', 'value')
            date_column = self.params.get('date_column', 'date')
            period = self.params.get('period', 12)
            model = self.params.get('model', 'additive')  # additive or multiplicative
            
            if value_column not in data.columns:
                self.logger.error(f"Colonne de valeurs '{value_column}' introuvable")
                return data
                
            if date_column not in data.columns:
                self.logger.error(f"Colonne de dates '{date_column}' introuvable")
                return data
                
            # Préparer les données
            ts_data = data[[date_column, value_column]].copy()
            ts_data[date_column] = pd.to_datetime(ts_data[date_column])
            ts_data = ts_data.sort_values(date_column)
            ts_data = ts_data.set_index(date_column)
            
            # Décomposition simple (sans statsmodels pour éviter les dépendances)
            # Tendance via moyenne mobile
            trend = ts_data[value_column].rolling(window=period).mean()
            
            if model == 'additive':
                # Modèle additif: y = trend + seasonal + residual
                detrended = ts_data[value_column] - trend
                seasonal = detrended.groupby(detrended.index.dayofyear % period).transform('mean')
                residual = detrended - seasonal
            else:
                # Modèle multiplicatif: y = trend * seasonal * residual
                detrended = ts_data[value_column] / trend
                seasonal = detrended.groupby(detrended.index.dayofyear % period).transform('mean')
                residual = detrended / seasonal
                
            # Ajouter les composantes au DataFrame
            result_data = data.copy()
            
            # Réaligner avec l'index original
            for i, idx in enumerate(data.index):
                if i < len(trend):
                    if pd.notna(trend.iloc[i]):
                        result_data.loc[idx, 'ts_trend'] = trend.iloc[i]
                    if pd.notna(seasonal.iloc[i]):
                        result_data.loc[idx, 'ts_seasonal'] = seasonal.iloc[i]
                    if pd.notna(residual.iloc[i]):
                        result_data.loc[idx, 'ts_residual'] = residual.iloc[i]
            
            self.logger.info(f"Décomposition temporelle terminée (modèle {model})")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur décomposition temporelle : {str(e)}")
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
BlockRegistry.register_block('TimeSeriesDecompositionBlock', TimeSeriesDecompositionBlock)
