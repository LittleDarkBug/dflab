from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class KMeansClustering(BlockBase):
    """
    Bloc de clustering K-Means avec optimisation du nombre de clusters.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="KMeansClustering", category="unsupervised", params=params)
        self.logger = get_logger("KMeansClustering")
        self.model = None

    def fit(self, X, y=None):
        n_clusters = self.params.get("n_clusters", 8)
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(X)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("KMeans non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.predict(X), index=X.index, name="cluster")
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute K-Means clustering."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            n_clusters = self.params.get('n_clusters', 8)
            auto_optimize = self.params.get('auto_optimize', False)
            columns_to_use = self.params.get('columns')
            
            # Sélectionner les colonnes numériques
            if columns_to_use:
                if isinstance(columns_to_use, str):
                    columns_to_use = [col.strip() for col in columns_to_use.split(',')]
                numeric_data = data[columns_to_use].select_dtypes(include=[np.number])
            else:
                numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                self.logger.warning("Aucune donnée numérique trouvée pour le clustering")
                return data
            
            # Supprimer les lignes avec des valeurs manquantes
            clean_data = numeric_data.dropna()
            if len(clean_data) == 0:
                self.logger.warning("Aucune donnée valide après suppression des valeurs manquantes")
                return data
            
            # Optimisation automatique du nombre de clusters (méthode du coude)
            if auto_optimize:
                max_k = min(10, len(clean_data) - 1)
                silhouette_scores = []
                
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(clean_data)
                    silhouette_avg = silhouette_score(clean_data, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                # Choisir le k avec le meilleur score de silhouette
                best_k = np.argmax(silhouette_scores) + 2
                n_clusters = best_k
                self.logger.info(f"Nombre optimal de clusters trouvé: {n_clusters}")
            
            # Appliquer K-Means
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.model.fit_predict(clean_data)
            
            # Calculer le score de silhouette
            silhouette_avg = silhouette_score(clean_data, cluster_labels)
            
            # Ajouter les clusters au DataFrame
            result_data = data.copy()
            result_data['kmeans_cluster'] = -1  # -1 pour les données non classifiées
            result_data.loc[clean_data.index, 'kmeans_cluster'] = cluster_labels
            
            # Ajouter les centres de clusters comme information
            centers = self.model.cluster_centers_
            for i, center in enumerate(centers):
                result_data[f'distance_to_cluster_{i}'] = np.nan
                if len(clean_data) > 0:
                    distances = np.linalg.norm(clean_data.values - center, axis=1)
                    result_data.loc[clean_data.index, f'distance_to_cluster_{i}'] = distances
            
            self.logger.info(f"K-Means clustering terminé: {n_clusters} clusters")
            self.logger.info(f"Score de silhouette: {silhouette_avg:.4f}")
            self.logger.info(f"Inertie: {self.model.inertia_:.2f}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du clustering K-Means : {str(e)}")
            return data
        
    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Méthode process pour compatibilité avec BlockBase.
        """
        return self.execute(data, **kwargs)


# Auto-enregistrement du bloc
BlockRegistry.register_block('KMeansClustering', KMeansClustering)
