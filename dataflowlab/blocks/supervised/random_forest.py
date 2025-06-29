"""
Bloc Random Forest pour classification et régression.
"""

from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class RandomForestBlock(BlockBase):
    """
    Bloc Random Forest pour classification et régression.
    
    Paramètres configurables:
    - task_type: 'classification' ou 'regression'
    - target_column: nom de la colonne cible
    - n_estimators: nombre d'arbres
    - max_depth: profondeur maximale
    - random_state: graine aléatoire
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'task_type': 'classification',
            'target_column': None,
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'max_features': 'sqrt'
        }
        super().__init__(
            name="RandomForest",
            category="supervised",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.feature_names_ = None
        self.target_name_ = None
        
    def get_config_interface(self) -> Dict[str, Any]:
        """Configuration interface pour l'UI."""
        return {
            'task_type': {
                'type': 'select',
                'options': ['classification', 'regression'],
                'default': 'classification',
                'description': 'Type de tâche'
            },
            'target_column': {
                'type': 'text',
                'default': '',
                'description': 'Nom de la colonne cible'
            },
            'n_estimators': {
                'type': 'number',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Nombre d\'arbres'
            },
            'max_depth': {
                'type': 'number',
                'default': 10,
                'min': 1,
                'max': 50,
                'description': 'Profondeur maximale (vide = illimitée)'
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
            
        return True, "OK"
    
    def process(self, data: pd.DataFrame = None, **kwargs) -> Dict[str, Any]:
        """Implémentation de process pour compatibilité."""
        return self._execute_impl({'data': data, **kwargs})
    
    def _execute_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute Random Forest."""
        try:
            # Validation
            is_valid, msg = self.validate_inputs(inputs)
            if not is_valid:
                raise ValueError(f"Validation échouée: {msg}")
                
            data = inputs['data'].copy()
            target_col = self.params['target_column']
            task_type = self.params['task_type']
            
            self.logger.info(f"Démarrage Random Forest ({task_type}) sur {len(data)} échantillons")
            
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
            
            # Initialiser le modèle selon le type de tâche
            if task_type == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=self.params['n_estimators'],
                    max_depth=self.params['max_depth'],
                    max_features=self.params['max_features'],
                    random_state=self.params['random_state']
                )
            else:  # regression
                self.model = RandomForestRegressor(
                    n_estimators=self.params['n_estimators'],
                    max_depth=self.params['max_depth'],
                    max_features=self.params['max_features'],
                    random_state=self.params['random_state']
                )
                
            # Entraîner le modèle
            self.model.fit(X_clean, y_clean)
            
            # Prédictions
            y_pred = self.model.predict(X_clean)
            
            # Métriques selon le type de tâche
            if task_type == 'classification':
                accuracy = accuracy_score(y_clean, y_pred)
                metrics = {'accuracy': accuracy, 'n_samples': len(X_clean), 'n_features': len(self.feature_names_)}
            else:  # regression
                r2 = r2_score(y_clean, y_pred)
                rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
                metrics = {'r2_score': r2, 'rmse': rmse, 'n_samples': len(X_clean), 'n_features': len(self.feature_names_)}
            
            # Ajouter prédictions au DataFrame
            result_data = data.copy()
            result_data['predicted'] = np.nan
            result_data.loc[mask, 'predicted'] = y_pred
            
            # Importance des features
            feature_importance = dict(zip(self.feature_names_, self.model.feature_importances_))
            
            # Résultat
            result = {
                'data': result_data,
                'model': self.model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'info': {
                    'task_type': task_type,
                    'feature_names': self.feature_names_,
                    'target_name': self.target_name_
                }
            }
            
            self.logger.info(f"Random Forest terminé")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {str(e)}")
            raise


# Auto-enregistrement du bloc
BlockRegistry.register_block('RandomForestBlock', RandomForestBlock)
