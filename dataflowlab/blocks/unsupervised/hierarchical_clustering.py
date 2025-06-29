from typing import Any, Dict, Optional
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class HierarchicalClustering(BlockBase):
    """
    Bloc de clustering hiérarchique.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="HierarchicalClustering", params=params, category="unsupervised")
        self.logger = get_logger("HierarchicalClustering")
        self.model = None

    def fit(self, X, y=None):
        n_clusters = self.params.get("n_clusters", 2)
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
        self.model.fit(X)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Clustering non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.fit_predict(X), index=X.index, name="cluster")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute hierarchical clustering."""
        try:
            n_clusters = self.params.get('n_clusters', 2)
            
            # Sélection des colonnes numériques
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                self.logger.error("Aucune colonne numérique trouvée pour le clustering")
                return data
            
            X = data[numeric_cols].fillna(0)  # Simple imputation
            
            # Clustering hiérarchique
            model = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(X)
            
            # Ajouter les résultats
            result_data = data.copy()
            result_data['cluster'] = cluster_labels
            
            self.logger.info(f"Clustering hiérarchique terminé - {n_clusters} clusters")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du clustering hiérarchique : {str(e)}")
            return data

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Méthode process pour compatibilité avec BlockBase.
        """
        return self.execute(data, **kwargs)

# Auto-enregistrement du bloc
BlockRegistry.register_block('HierarchicalClustering', HierarchicalClustering)
