from typing import Any, Dict, Optional
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class SHAPExplainerBlock(BlockBase):
    """
    Bloc d'explication SHAP.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="SHAPExplainer", params=params, category="evaluation")
        self.logger = get_logger("SHAPExplainer")
        self.explainer = None

    def fit(self, X, y=None):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP n'est pas installé. Installez-le avec: pip install shap")
        model = self.params.get("model")
        self.explainer = shap.Explainer(model, X)
        return self

    def transform(self, X):
        if self.explainer is None:
            raise RuntimeError("Explainer non initialisé. Appelez fit d'abord.")
        return self.explainer(X)


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
BlockRegistry.register_block('SHAPExplainerBlock', SHAPExplainerBlock)
