from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class RegressionMetrics(BlockBase):
    """
    Bloc de calcul des métriques de régression.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="RegressionMetrics", category="evaluation", params=params)
        self.logger = get_logger("RegressionMetrics")

    def transform(self, X):
        y_true = self.params.get("y_true")
        y_pred = self.params.get("y_pred")
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        return metrics
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute regression metrics calculation."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            true_column = self.params.get('true_column', 'y_true')
            pred_column = self.params.get('pred_column', 'y_pred')
            
            if true_column not in data.columns:
                self.logger.error(f"Colonne vraies valeurs '{true_column}' non trouvée")
                return data
            
            if pred_column not in data.columns:
                self.logger.error(f"Colonne prédictions '{pred_column}' non trouvée")
                return data
            
            # Extraire les valeurs vraies et prédites
            mask = data[true_column].notna() & data[pred_column].notna()
            y_true = data.loc[mask, true_column]
            y_pred = data.loc[mask, pred_column]
            
            if len(y_true) == 0:
                self.logger.warning("Aucune donnée valide pour calculer les métriques")
                return data
            
            # Calculer les métriques de régression
            metrics = {}
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error) si pas de zéros
            try:
                if not np.any(y_true == 0):
                    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
                else:
                    self.logger.warning("MAPE non calculable (valeurs vraies contiennent des zéros)")
                    metrics['mape'] = None
            except Exception as e:
                self.logger.warning(f"Impossible de calculer MAPE: {str(e)}")
                metrics['mape'] = None
            
            # Calculer des métriques additionnelles
            residuals = y_true - y_pred
            metrics['mean_residual'] = np.mean(residuals)
            metrics['std_residual'] = np.std(residuals)
            
            # Log des métriques
            self.logger.info("Métriques de régression calculées:")
            for metric, value in metrics.items():
                if value is not None:
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            # Ajouter les métriques au DataFrame comme nouvelles colonnes
            result_data = data.copy()
            for metric, value in metrics.items():
                if value is not None:
                    result_data[f'metric_{metric}'] = value
            
            # Ajouter les résidus
            result_data['residuals'] = np.nan
            result_data.loc[mask, 'residuals'] = residuals
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de régression : {str(e)}")
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
BlockRegistry.register_block('RegressionMetrics', RegressionMetrics)
