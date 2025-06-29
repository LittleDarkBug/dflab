from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class PCATransformer(BlockBase):
    """
    Bloc de réduction de dimensionnalité par PCA.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="PCATransformer", category="feature_engineering", params=params)
        self.logger = get_logger("PCATransformer")
        self.pca = None

    def fit(self, X, y=None):
        n_components = self.params.get("n_components", 2)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)
        return self

    def transform(self, X):
        if self.pca is None:
            raise RuntimeError("PCA non initialisé. Appelez fit d'abord.")
        arr = self.pca.transform(X)
        columns = [f"PC{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns, index=X.index)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute PCA transformation."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            n_components = self.params.get('n_components', 2)
            columns_to_use = self.params.get('columns')
            
            # Sélectionner les colonnes numériques
            if columns_to_use:
                if isinstance(columns_to_use, str):
                    columns_to_use = [col.strip() for col in columns_to_use.split(',')]
                numeric_data = data[columns_to_use].select_dtypes(include=[np.number])
            else:
                numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                self.logger.warning("Aucune donnée numérique trouvée pour PCA")
                return data
            
            # Supprimer les lignes avec des valeurs manquantes
            clean_data = numeric_data.dropna()
            if len(clean_data) == 0:
                self.logger.warning("Aucune donnée valide après suppression des valeurs manquantes")
                return data
            
            # Ajuster n_components si nécessaire
            max_components = min(clean_data.shape[0], clean_data.shape[1])
            if n_components > max_components:
                n_components = max_components
                self.logger.warning(f"n_components ajusté à {n_components}")
            
            # Appliquer PCA
            self.pca = PCA(n_components=n_components)
            pca_result = self.pca.fit_transform(clean_data)
            
            # Créer les noms de colonnes
            pca_columns = [f"PC{i+1}" for i in range(n_components)]
            
            # Créer le DataFrame résultat
            pca_df = pd.DataFrame(
                pca_result,
                columns=pca_columns,
                index=clean_data.index
            )
            
            # Combiner avec les données originales (en excluant les colonnes utilisées pour PCA)
            result_data = data.copy()
            
            # Supprimer les colonnes originales utilisées pour PCA
            if columns_to_use:
                result_data = result_data.drop(columns=columns_to_use, errors='ignore')
            else:
                result_data = result_data.select_dtypes(exclude=[np.number])
            
            # Ajouter les composantes principales
            for col in pca_columns:
                result_data[col] = np.nan
                result_data.loc[clean_data.index, col] = pca_df[col]
            
            # Log des informations sur la variance expliquée
            variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratio)
            
            self.logger.info(f"PCA appliquée: {n_components} composantes")
            self.logger.info(f"Variance expliquée: {cumulative_variance[-1]:.4f}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'application de PCA : {str(e)}")
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
BlockRegistry.register_block('PCATransformer', PCATransformer)
