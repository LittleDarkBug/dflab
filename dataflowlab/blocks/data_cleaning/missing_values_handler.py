from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
# Import conditionnel d'IterativeImputer pour éviter le warning
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_AVAILABLE = True
except ImportError:
    ITERATIVE_AVAILABLE = False
    IterativeImputer = None

from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class MissingValuesHandler(BlockBase):
    """
    Bloc d'imputation des valeurs manquantes avec différentes stratégies.
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "MissingValuesHandler", params=params, category="data_cleaning")
        self.imputer = None
        self.numeric_columns = []
        self.categorical_columns = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'MissingValuesHandler':
        """Ajuste l'imputer sur les données."""
        try:
            if X is None or X.empty:
                raise ValueError("Aucune donnée pour l'ajustement")
            
            # Identification des colonnes par type
            self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            strategy = self.params.get("strategy", "mean")
            n_neighbors = self.params.get("n_neighbors", 5)
            max_iter = self.params.get("max_iter", 10)
            
            # Sélection de l'imputer selon la stratégie
            if strategy == "mean":
                self.imputer = SimpleImputer(strategy="mean")
            elif strategy == "median":
                self.imputer = SimpleImputer(strategy="median")
            elif strategy == "mode":
                self.imputer = SimpleImputer(strategy="most_frequent")
            elif strategy == "constant":
                fill_value = self.params.get("fill_value", 0)
                self.imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
            elif strategy == "knn":
                self.imputer = KNNImputer(n_neighbors=n_neighbors)
            elif strategy == "iterative":
                if not ITERATIVE_AVAILABLE:
                    raise ValueError("IterativeImputer n'est pas disponible. Utilisez 'mean', 'median', 'mode', 'knn' ou 'constant'.")
                self.imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            else:
                raise ValueError(f"Stratégie inconnue: {strategy}")
            
            # Ajustement selon le type de données
            if strategy in ["mean", "median"] and self.numeric_columns:
                # Pour mean/median, on ne traite que les colonnes numériques
                self.imputer.fit(X[self.numeric_columns])
            elif strategy == "mode":
                # Mode peut s'appliquer à tous types
                self.imputer.fit(X)
            elif strategy in ["knn", "iterative"]:
                # KNN et iterative nécessitent des données numériques
                if not self.numeric_columns:
                    raise ValueError(f"Stratégie {strategy} nécessite des colonnes numériques")
                self.imputer.fit(X[self.numeric_columns])
            else:
                self.imputer.fit(X)
            
            self._fitted = True
            logger.info(f"MissingValuesHandler ajusté avec stratégie: {strategy}")
            
            return self
            
        except Exception as e:
            error_msg = f"Erreur lors de l'ajustement de l'imputer: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Impute les valeurs manquantes selon la stratégie configurée.
        
        Args:
            data: DataFrame avec valeurs manquantes
            
        Returns:
            DataFrame avec valeurs imputées
        """
        try:
            if data is None or data.empty:
                raise ValueError("Aucune donnée à traiter")
            
            if not self._fitted:
                # Auto-fit si pas encore fait
                self.fit(data)
            
            result = data.copy()
            strategy = self.params.get("strategy", "mean")
            
            # Application de l'imputation selon la stratégie
            if strategy in ["mean", "median"] and self.numeric_columns:
                # Imputation des colonnes numériques seulement
                numeric_data = self.imputer.transform(data[self.numeric_columns])
                result[self.numeric_columns] = numeric_data
                
                # Mode pour les colonnes catégorielles si nécessaire
                if self.categorical_columns:
                    mode_imputer = SimpleImputer(strategy="most_frequent")
                    categorical_data = mode_imputer.fit_transform(data[self.categorical_columns])
                    result[self.categorical_columns] = categorical_data
                    
            elif strategy == "mode":
                # Mode pour toutes les colonnes
                imputed_data = self.imputer.transform(data)
                result = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
                
            elif strategy in ["knn", "iterative"]:
                # Imputation des colonnes numériques
                if self.numeric_columns:
                    numeric_data = self.imputer.transform(data[self.numeric_columns])
                    result[self.numeric_columns] = numeric_data
                
                # Mode pour les colonnes catégorielles
                if self.categorical_columns:
                    mode_imputer = SimpleImputer(strategy="most_frequent")
                    categorical_data = mode_imputer.fit_transform(data[self.categorical_columns])
                    result[self.categorical_columns] = categorical_data
            else:
                # Stratégie constante ou autre
                imputed_data = self.imputer.transform(data)
                result = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)
            
            # Statistiques d'imputation
            missing_before = data.isnull().sum().sum()
            missing_after = result.isnull().sum().sum()
            
            self._output_data = result
            logger.info(f"Imputation terminée: {missing_before} valeurs manquantes -> {missing_after}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur lors de l'imputation: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "strategy",
                "type": "select",
                "label": "Stratégie d'imputation",
                "default": "mean",
                "options": ["mean", "median", "mode", "constant", "knn", "iterative"],
                "required": True
            },
            {
                "name": "fill_value",
                "type": "number",
                "label": "Valeur constante (si strategy=constant)",
                "default": 0
            },
            {
                "name": "n_neighbors",
                "type": "number",
                "label": "Nombre de voisins (KNN)",
                "default": 5,
                "min": 1,
                "max": 20
            },
            {
                "name": "max_iter",
                "type": "number",
                "label": "Iterations max (Iterative)",
                "default": 10,
                "min": 1,
                "max": 100
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('MissingValuesHandler', MissingValuesHandler)
