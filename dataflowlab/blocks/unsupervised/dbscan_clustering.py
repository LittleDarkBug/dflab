"""
Bloc DBSCAN pour le clustering par densité.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class DBSCANBlock(BlockBase):
    """
    Bloc de clustering DBSCAN (Density-Based Spatial Clustering).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'eps': 0.5,  # Distance maximale entre deux échantillons
            'min_samples': 5,  # Nombre minimum d'échantillons dans un voisinage
            'scale_features': True,  # Normaliser les features
            'columns': []  # Colonnes spécifiques (vide = toutes les numériques)
        }
        super().__init__(
            name="DBSCAN",
            category="unsupervised",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.scaler = None
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute DBSCAN clustering."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            eps = self.params.get('eps', 0.5)
            min_samples = self.params.get('min_samples', 5)
            scale_features = self.params.get('scale_features', True)
            columns = self.params.get('columns', [])
            
            # Sélectionner les colonnes
            if columns:
                numeric_cols = [col for col in columns if col in data.columns]
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                self.logger.warning("Aucune colonne numérique trouvée")
                return data
            
            # Préparer les données
            X = data[numeric_cols].copy()
            
            # Supprimer les lignes avec des valeurs manquantes
            mask = ~X.isnull().any(axis=1)
            X_clean = X[mask]
            
            if len(X_clean) == 0:
                self.logger.warning("Aucune donnée valide après nettoyage")
                return data
            
            # Normalisation optionnelle
            if scale_features:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X_clean)
            else:
                X_scaled = X_clean.values
            
            # Clustering DBSCAN
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = self.model.fit_predict(X_scaled)
            
            # Ajouter les résultats au DataFrame
            result_data = data.copy()
            result_data['dbscan_cluster'] = -1  # Valeur par défaut pour les lignes manquantes
            result_data.loc[mask, 'dbscan_cluster'] = cluster_labels
            
            # Statistiques
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            self.logger.info(f"DBSCAN terminé - Clusters: {n_clusters}, Noise: {n_noise}, Eps: {eps}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du clustering DBSCAN : {str(e)}")
            return data

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Méthode process pour compatibilité avec BlockBase.
        """
        return self.execute(data, **kwargs)


# Auto-enregistrement du bloc
BlockRegistry.register_block('DBSCANBlock', DBSCANBlock)
