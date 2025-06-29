"""
Bloc de régression logistique pour la classification.
"""

from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class LogisticRegressionBlock(BlockBase):
    """
    Bloc de régression logistique pour la classification.
    
    Paramètres configurables:
    - target_column: nom de la colonne cible
    - max_iter: nombre maximum d'itérations
    - C: inverse du paramètre de régularisation
    - solver: algorithme d'optimisation
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'target_column': None,
            'max_iter': 1000,
            'C': 1.0,
            'solver': 'lbfgs',
            'random_state': 42
        }
        super().__init__(
            name="LogisticRegression",
            category="supervised",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.feature_names_ = None
        self.target_name_ = None
        self.classes_ = None
        
    def get_config_interface(self) -> Dict[str, Any]:
        """Configuration interface pour l'UI."""
        return {
            'target_column': {
                'type': 'text',
                'default': '',
                'description': 'Nom de la colonne cible'
            },
            'max_iter': {
                'type': 'number',
                'default': 1000,
                'min': 100,
                'max': 10000,
                'description': 'Nombre maximum d\'itérations'
            },
            'C': {
                'type': 'number',
                'default': 1.0,
                'min': 0.001,
                'max': 100.0,
                'description': 'Inverse du paramètre de régularisation'
            },
            'solver': {
                'type': 'select',
                'options': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'default': 'lbfgs',
                'description': 'Algorithme d\'optimisation'
            }
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, str]:
        """Valide les données d'entrée."""
        if 'data' not in inputs:
            return False, "Données d'entrée manquantes"
            
        data = inputs['data']
        if not isinstance(data, pd.DataFrame):
            return False, "Les données doivent être un DataFrame pandas"
            
        if len(data) == 0:
            return False, "DataFrame vide"
            
        target_col = self.params.get('target_column')
        if not target_col:
            return False, "Colonne cible non spécifiée"
            
        if target_col not in data.columns:
            return False, f"Colonne cible '{target_col}' introuvable"
            
        # Vérifier qu'il y a au moins une colonne numérique pour les features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if len(numeric_cols) == 0:
            return False, "Aucune colonne numérique trouvée pour les features"
            
        return True, "OK"
    
    def process(self, data: pd.DataFrame = None, **kwargs) -> Dict[str, Any]:
        """Implémentation de la méthode abstraite process()."""
        return self._execute_impl({'data': data, **kwargs})
    
    def _execute_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute la régression logistique."""
        try:
            # Validation
            is_valid, msg = self.validate_inputs(inputs)
            if not is_valid:
                raise ValueError(f"Validation échouée: {msg}")
                
            data = inputs['data'].copy()
            target_col = self.params['target_column']
            
            self.logger.info(f"Démarrage régression logistique sur {len(data)} échantillons")
            
            # Préparer les données
            y = data[target_col]
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
            
            # Supprimer les lignes avec des valeurs manquantes
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                raise ValueError("Aucune donnée valide après nettoyage")
                
            self.feature_names_ = list(X_clean.columns)
            self.target_name_ = target_col
            self.classes_ = sorted(y_clean.unique())
            
            # Initialiser le modèle
            self.model = LogisticRegression(
                max_iter=self.params['max_iter'],
                C=self.params['C'],
                solver=self.params['solver'],
                random_state=self.params['random_state']
            )
                
            # Entraîner le modèle
            self.model.fit(X_clean, y_clean)
            
            # Prédictions
            y_pred = self.model.predict(X_clean)
            y_pred_proba = self.model.predict_proba(X_clean)
            
            # Métriques
            accuracy = accuracy_score(y_clean, y_pred)
            
            # Ajouter prédictions au DataFrame
            result_data = data.copy()
            result_data['predicted'] = np.nan
            result_data.loc[mask, 'predicted'] = y_pred
            
            # Ajouter probabilités pour chaque classe
            for i, class_name in enumerate(self.classes_):
                col_name = f'proba_{class_name}'
                result_data[col_name] = np.nan
                result_data.loc[mask, col_name] = y_pred_proba[:, i]
            
            # Résultat
            result = {
                'data': result_data,
                'model': self.model,
                'metrics': {
                    'accuracy': accuracy,
                    'n_samples': len(X_clean),
                    'n_features': len(self.feature_names_),
                    'n_classes': len(self.classes_)
                },
                'classification_report': classification_report(y_clean, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_clean, y_pred).tolist(),
                'info': {
                    'feature_names': self.feature_names_,
                    'target_name': self.target_name_,
                    'classes': self.classes_,
                    'C': self.params['C'],
                    'solver': self.params['solver']
                }
            }
            
            self.logger.info(f"Classification terminée - Accuracy = {accuracy:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {str(e)}")
            raise


# Auto-enregistrement du bloc
BlockRegistry.register_block('LogisticRegressionBlock', LogisticRegressionBlock)
