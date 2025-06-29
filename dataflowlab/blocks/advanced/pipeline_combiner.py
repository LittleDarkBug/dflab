from typing import Any, Dict, Optional, List
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class PipelineCombiner(BlockBase):
    """
    Bloc de combinaison de plusieurs pipelines.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="PipelineCombiner", params=params, category="advanced")
        self.logger = get_logger("PipelineCombiner")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute pipeline combination."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors de la combinaison de pipelines : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Méthode process pour compatibilité avec BlockBase."""
        return self.execute(data, **kwargs)

    def transform(self, X):
        pipelines: List[BlockBase] = self.params.get("pipelines", [])
        results = [pipe.transform(X) for pipe in pipelines]
        return results

# Auto-enregistrement du bloc
BlockRegistry.register_block('PipelineCombiner', PipelineCombiner)
