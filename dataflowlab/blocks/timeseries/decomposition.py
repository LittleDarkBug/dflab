"""
Bloc Time Series Decomposition pour la décomposition de séries temporelles.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class TimeSeriesDecompositionBlock(BlockBase):
    """
    Bloc de décomposition de séries temporelles (tendance, saisonnalité, résidus).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'time_column': None,
            'value_column': None,
            'model': 'additive',  # 'additive' ou 'multiplicative'
            'period': None,  # Période saisonnière (auto-détection si None)
            'extrapolate_trend': 'freq'
        }
        super().__init__(
            name="TimeSeriesDecomposition",
            category="timeseries",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.decomposition_result = None
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute time series decomposition."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            time_column = self.params.get('time_column')
            value_column = self.params.get('value_column')
            model = self.params.get('model', 'additive')
            period = self.params.get('period')
            
            if not time_column or time_column not in data.columns:
                self.logger.error(f"Colonne temporelle '{time_column}' non trouvée")
                return data
                
            if not value_column or value_column not in data.columns:
                self.logger.error(f"Colonne de valeurs '{value_column}' non trouvée")
                return data
            
            # Préparer les données
            ts_data = data[[time_column, value_column]].copy()
            
            # Convertir la colonne temporelle
            ts_data[time_column] = pd.to_datetime(ts_data[time_column])
            ts_data = ts_data.sort_values(time_column)
            ts_data.set_index(time_column, inplace=True)
            
            # Supprimer les valeurs manquantes
            ts_data = ts_data.dropna()
            
            if len(ts_data) < 10:
                self.logger.warning("Pas assez de données pour la décomposition")
                return data
            
            # Décomposition simple (sans statsmodels pour éviter les dépendances)
            series = ts_data[value_column]
            
            # Calcul de la tendance (moyenne mobile)
            window_size = min(len(series) // 4, 12) if period is None else period
            trend = series.rolling(window=window_size, center=True).mean()
            
            # Détrended series
            if model == 'additive':
                detrended = series - trend
            else:  # multiplicative
                detrended = series / trend
            
            # Saisonnalité (moyenne des résidus par période)
            if period is None:
                period = 12  # Valeur par défaut
            
            seasonal = detrended.groupby(detrended.index.dayofyear % period).transform('mean')
            
            # Résidus
            if model == 'additive':
                residual = series - trend - seasonal
            else:
                residual = series / (trend * seasonal)
            
            # Créer le DataFrame résultat
            result_data = data.copy()
            
            # Ajouter les composantes décomposées
            for idx, (t, tr, s, r) in zip(ts_data.index, trend, seasonal, residual):
                mask = data[time_column] == t
                if mask.any():
                    result_data.loc[mask, 'ts_trend'] = tr
                    result_data.loc[mask, 'ts_seasonal'] = s
                    result_data.loc[mask, 'ts_residual'] = r
            
            self.logger.info(f"Décomposition temporelle terminée - Modèle: {model}, Période: {period}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la décomposition temporelle : {str(e)}")
            return data


# Auto-enregistrement du bloc
BlockRegistry.register_block('TimeSeriesDecompositionBlock', TimeSeriesDecompositionBlock)
