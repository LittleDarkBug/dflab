from typing import Any, Dict, Optional, Callable
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class CustomCodeBlock(BlockBase):
    """
    Bloc d'exécution de code Python personnalisé.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="CustomCodeBlock", params=params, category="advanced")
        self.logger = get_logger("CustomCodeBlock")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute custom code."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution du code personnalisé : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Méthode process pour compatibilité avec BlockBase."""
        return self.execute(data, **kwargs)

    def transform(self, X):
        func: Callable = self.params.get("func")
        if not callable(func):
            raise ValueError("Le paramètre 'func' doit être une fonction Python.")
        return func(X)

# Auto-enregistrement du bloc
BlockRegistry.register_block('CustomCodeBlock', CustomCodeBlock)
