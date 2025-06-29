from typing import Any, Dict, Optional
import pandas as pd
# Import conditionnel de statsmodels
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    seasonal_decompose = None

from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class SeasonalityDecomposer(BlockBase):
    """
    Bloc de décomposition tendance/saisonnalité.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="SeasonalityDecomposer", params=params, category="timeseries")
        self.logger = get_logger("SeasonalityDecomposer")

    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute seasonality decomposition."""
        if not STATSMODELS_AVAILABLE:
            self.logger.error("statsmodels n'est pas disponible. Impossible de faire la décomposition saisonnière.")
            return {"error": "statsmodels not available"}
        
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de la décomposition saisonnière : {str(e)}")
            return {"error": str(e)}

    def transform(self, X):
        col = self.params.get("col")
        period = self.params.get("period", 12)
        result = seasonal_decompose(X[col], period=period)
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid
        }

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data by applying seasonality decomposition."""
        if not STATSMODELS_AVAILABLE:
            self.logger.error("statsmodels n'est pas disponible. Impossible de faire la décomposition saisonnière.")
            raise ImportError("statsmodels package is required for seasonality decomposition")
        
        try:
            result = self.transform(data)
            
            # Convertir le résultat en DataFrame
            if isinstance(result, dict):
                # Créer un DataFrame avec les composantes décomposées
                decomposed_df = pd.DataFrame({
                    'trend': result['trend'],
                    'seasonal': result['seasonal'], 
                    'residual': result['resid']
                })
                # Supprimer les valeurs NaN
                decomposed_df = decomposed_df.dropna()
                return decomposed_df
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la décomposition saisonnière : {str(e)}")
            raise e

# Auto-enregistrement du bloc
BlockRegistry.register_block('SeasonalityDecomposer', SeasonalityDecomposer)
