from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class DecisionTreeBlock(BlockBase):
    """
    Bloc arbre de décision (classification ou régression).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="DecisionTree", category="supervised", params=params)
        self.logger = get_logger("DecisionTree")
        self.model = None

    def fit(self, X, y):
        task = self.params.get("task", "classification")
        if task == "regression":
            self.model = DecisionTreeRegressor()
        else:
            self.model = DecisionTreeClassifier()
        self.model.fit(X, y)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.predict(X), index=X.index, name="prediction")
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute decision tree training and prediction."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            target_column = self.params.get("target_column", "target")
            task_type = self.params.get("task_type", "classification")
            max_depth = self.params.get("max_depth", None)
            
            if target_column not in data.columns:
                self.logger.error(f"Colonne cible '{target_column}' non trouvée")
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
                
            # Initialiser le modèle selon le type de tâche
            if task_type == "regression":
                self.model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                self.model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=42
                )
                
            # Entraîner le modèle
            self.model.fit(X_clean, y_clean)
            
            # Prédictions
            y_pred = self.model.predict(X_clean)
            
            # Ajouter prédictions au DataFrame
            result_data = data.copy()
            result_data['dt_prediction'] = np.nan
            result_data.loc[mask, 'dt_prediction'] = y_pred
            
            # Calculer les métriques
            if task_type == "regression":
                r2 = r2_score(y_clean, y_pred)
                self.logger.info(f"Decision Tree R² score: {r2:.4f}")
            else:
                accuracy = accuracy_score(y_clean, y_pred)
                self.logger.info(f"Decision Tree accuracy: {accuracy:.4f}")
            
            self.logger.info(f"Decision Tree entraîné avec {len(X_clean)} échantillons")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de Decision Tree : {str(e)}")
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
BlockRegistry.register_block('DecisionTreeBlock', DecisionTreeBlock)
