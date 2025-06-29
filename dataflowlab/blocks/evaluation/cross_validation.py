from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class CrossValidationBlock(BlockBase):
    """
    Bloc de validation croisée avec métriques.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="CrossValidation", params=params)
        self.logger = get_logger("CrossValidation")

    def transform(self, X):
        estimator = self.params.get("estimator")
        X_data = self.params.get("X")
        y_data = self.params.get("y")
        scoring = self.params.get("scoring", "accuracy")
        cv = self.params.get("cv", 5)
        scores = cross_val_score(estimator, X_data, y_data, scoring=scoring, cv=cv)
        return scores
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute cross validation."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            target_column = self.params.get('target_column')
            cv_folds = self.params.get('cv_folds', 5)
            
            if not target_column or target_column not in data.columns:
                self.logger.error(f"Colonne cible '{target_column}' non trouvée")
                return data
            
            # Simple validation croisée
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            
            y = data[target_column]
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
            
            model = RandomForestClassifier(random_state=42)
            scores = cross_val_score(model, X, y, cv=cv_folds)
            
            result_data = data.copy()
            result_data.attrs['cv_scores'] = scores
            result_data.attrs['cv_mean'] = scores.mean()
            
            self.logger.info(f"CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation croisée : {str(e)}")
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
BlockRegistry.register_block('CrossValidationBlock', CrossValidationBlock)
