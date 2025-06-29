"""
Bloc Polynomial Features pour générer des features polynomiales.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures as SkPolynomialFeatures
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class PolynomialFeaturesBlock(BlockBase):
    """
    Bloc de génération de features polynomiales.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'degree': 2,
            'include_bias': False,
            'interaction_only': False,
            'columns': []
        }
        super().__init__(
            name="PolynomialFeatures",
            category="feature_engineering",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.poly = None

    def fit(self, X, y=None):
        """Méthode fit pour compatibilité."""
        degree = self.params.get("degree", 2)
        self.poly = SkPolynomialFeatures(degree=degree, include_bias=False)
        self.poly.fit(X)
        return self

    def transform(self, X):
        """Méthode transform pour compatibilité."""
        if self.poly is None:
            raise RuntimeError("PolynomialFeatures non initialisé. Appelez fit d'abord.")
        arr = self.poly.transform(X)
        columns = self.poly.get_feature_names_out(X.columns)
        return pd.DataFrame(arr, columns=columns, index=X.index)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute polynomial features generation."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            degree = self.params.get('degree', 2)
            include_bias = self.params.get('include_bias', False)
            interaction_only = self.params.get('interaction_only', False)
            columns = self.params.get('columns', [])
            
            # Sélectionner les colonnes
            if columns:
                numeric_cols = [col for col in columns if col in data.columns]
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                self.logger.warning("Aucune colonne numérique trouvée")
                return data
            
            # Initialiser le transformateur
            self.poly = SkPolynomialFeatures(
                degree=degree,
                include_bias=include_bias,
                interaction_only=interaction_only
            )
            
            # Transformer les données
            X_numeric = data[numeric_cols]
            X_poly = self.poly.fit_transform(X_numeric)
            
            # Créer les noms des features
            feature_names = self.poly.get_feature_names_out(numeric_cols)
            
            # Créer le DataFrame résultat
            result_data = data.copy()
            
            # Ajouter les nouvelles features polynomiales
            for i, feature_name in enumerate(feature_names):
                if feature_name not in numeric_cols:
                    result_data[f'poly_{feature_name}'] = X_poly[:, i]
            
            self.logger.info(f"Features polynomiales générées: degré {degree}")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des features polynomiales : {str(e)}")
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
BlockRegistry.register_block('PolynomialFeaturesBlock', PolynomialFeaturesBlock)
