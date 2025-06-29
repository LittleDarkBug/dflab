"""
Bloc Ridge et Lasso Regression.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class RegularizedRegressionBlock(BlockBase):
    """
    Bloc de régression régularisée (Ridge, Lasso, ElasticNet).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'regression_type': 'ridge',  # ridge, lasso, elastic_net
            'target_column': None,
            'alpha': 1.0,  # Paramètre de régularisation
            'l1_ratio': 0.5,  # Ratio L1 pour ElasticNet
            'max_iter': 1000  # Nombre maximum d'itérations
        }
        super().__init__(
            name="RegularizedRegression",
            category="supervised",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute regularized regression."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            regression_type = self.params.get('regression_type', 'ridge')
            target_column = self.params.get('target_column')
            alpha = self.params.get('alpha', 1.0)
            l1_ratio = self.params.get('l1_ratio', 0.5)
            max_iter = self.params.get('max_iter', 1000)
            
            if not target_column or target_column not in data.columns:
                self.logger.error(f"Colonne cible '{target_column}' introuvable")
                return data
            
            # Préparer les données
            y = data[target_column]
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
            
            # Supprimer les lignes avec des valeurs manquantes
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                self.logger.warning("Aucune donnée valide après nettoyage")
                return data
            
            # Initialiser le modèle selon le type
            if regression_type == 'ridge':
                self.model = Ridge(alpha=alpha, max_iter=max_iter, random_state=42)
            elif regression_type == 'lasso':
                self.model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
            elif regression_type == 'elastic_net':
                self.model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    random_state=42
                )
            else:
                self.logger.error(f"Type de régression '{regression_type}' non supporté")
                return data
            
            # Entraîner le modèle
            self.model.fit(X_clean, y_clean)
            
            # Prédictions
            y_pred = self.model.predict(X_clean)
            
            # Ajouter prédictions au DataFrame
            result_data = data.copy()
            result_data[f'{regression_type}_prediction'] = np.nan
            result_data.loc[mask, f'{regression_type}_prediction'] = y_pred
            
            # Calculer les métriques
            r2 = r2_score(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
            
            self.logger.info(f"{regression_type.capitalize()} - R² score: {r2:.4f}, RMSE: {rmse:.4f}")
            
            # Ajouter coefficients si Ridge ou Lasso
            if hasattr(self.model, 'coef_'):
                coef_info = dict(zip(X_clean.columns, self.model.coef_))
                self.logger.info(f"Coefficients: {coef_info}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur régression régularisée : {str(e)}")
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
BlockRegistry.register_block('RegularizedRegressionBlock', RegularizedRegressionBlock)
