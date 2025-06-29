from typing import Any, Dict, Optional
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class AssociationRulesBlock(BlockBase):
    """
    Bloc d'extraction de règles d'association (FP-Growth).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="AssociationRules", params=params, category="unsupervised")
        self.logger = get_logger("AssociationRules")

    def transform(self, X):
        minSup = self.params.get("min_support", 0.5)
        minConf = self.params.get("min_confidence", 0.7)
        # X doit être un DataFrame binaire (transactions one-hot encoded)
        freq_itemsets = fpgrowth(X, min_support=minSup, use_colnames=True)
        rules = association_rules(freq_itemsets, metric="confidence", min_threshold=minConf)
        return rules

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute association rules mining."""
        try:
            # Supposer que les données sont déjà en format binaire one-hot
            rules = self.transform(data)
            
            self.logger.info(f"Extraction de règles d'association terminée - {len(rules)} règles trouvées")
            return rules
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de règles d'association : {str(e)}")
            return data

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Méthode process pour compatibilité avec BlockBase.
        """
        return self.execute(data, **kwargs)

# Auto-enregistrement du bloc
BlockRegistry.register_block('AssociationRulesBlock', AssociationRulesBlock)
