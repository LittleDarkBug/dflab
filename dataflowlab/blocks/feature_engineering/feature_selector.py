from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class FeatureSelector(BlockBase):
    """
    Bloc de sélection de features (univariée, RFE, SelectFromModel).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="FeatureSelector", params=params, category="feature_engineering")
        self.logger = get_logger("FeatureSelector")
        self.selector = None

    def fit(self, X, y=None):
        method = self.params.get("method", "univariate")
        k = self.params.get("k", 10)
        if method == "univariate":
            self.selector = SelectKBest(score_func=f_classif, k=k)
        elif method == "rfe":
            estimator = LogisticRegression()
            self.selector = RFE(estimator, n_features_to_select=k)
        elif method == "model":
            estimator = LogisticRegression()
            self.selector = SelectFromModel(estimator)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        if self.selector is None:
            raise RuntimeError("Selector non initialisé. Appelez fit d'abord.")
        arr = self.selector.transform(X)
        if hasattr(self.selector, 'get_support'):
            columns = X.columns[self.selector.get_support()]
        else:
            columns = X.columns[:arr.shape[1]]
        return pd.DataFrame(arr, columns=columns, index=X.index)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute feature selection."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            target_column = self.params.get('target_column')
            k_features = self.params.get('k_features', 10)
            
            if not target_column or target_column not in data.columns:
                self.logger.error(f"Colonne cible '{target_column}' non trouvée")
                return data
            
            # Préparer les données
            y = data[target_column]
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
            
            if X.empty:
                self.logger.warning("Aucune feature numérique trouvée")
                return data
            
            # Sélection simple basée sur la variance
            selector = VarianceThreshold()
            X_selected = selector.fit_transform(X)
            
            # Garder les k meilleures colonnes
            k_features = min(k_features, X_selected.shape[1])
            selected_cols = X.columns[selector.get_support()][:k_features]
            
            # Créer le DataFrame résultat
            result_data = data[list(selected_cols) + [target_column]]
            
            self.logger.info(f"Features sélectionnées: {len(selected_cols)} sur {X.shape[1]}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sélection de features : {str(e)}")
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
BlockRegistry.register_block('FeatureSelector', FeatureSelector)
