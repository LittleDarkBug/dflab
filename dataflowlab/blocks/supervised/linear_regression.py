"""
Bloc de régression linéaire avec variantes Ridge et Lasso.
"""

from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class LinearRegressionBlock(BlockBase):
    """
    Bloc de régression linéaire avec support pour Ridge et Lasso.
    
    Paramètres configurables:
    - regression_type: 'linear', 'ridge', 'lasso' 
    - alpha: facteur de régularisation pour Ridge/Lasso
    - target_column: nom de la colonne cible
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'regression_type': 'linear',
            'alpha': 1.0,
            'target_column': None,
            'fit_intercept': True,
            'normalize': False
        }
        super().__init__(
            name="LinearRegression",
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
            'regression_type': {
                'type': 'select',
                'options': ['linear', 'ridge', 'lasso'],
                'default': 'linear',
                'description': 'Type de régression'
            },
            'alpha': {
                'type': 'number',
                'default': 1.0,
                'min': 0.001,
                'max': 100.0,
                'description': 'Facteur de régularisation (Ridge/Lasso uniquement)'
            },
            'target_column': {
                'type': 'text',
                'default': '',
                'description': 'Nom de la colonne cible'
            },
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Ajuster l\'ordonnée à l\'origine'
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
        """Exécute la régression linéaire."""
        try:
            # Validation
            is_valid, msg = self.validate_inputs(inputs)
            if not is_valid:
                raise ValueError(f"Validation échouée: {msg}")
                
            data = inputs['data'].copy()
            target_col = self.params['target_column']
            
            self.logger.info(f"Démarrage régression linéaire sur {len(data)} échantillons")
            
            # Préparer les données
            y = data[target_col]
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_col])
            
            # Supprimer les lignes avec des valeurs manquantes
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                raise ValueError("Aucune donnée valide après nettoyage")
                
            self.feature_names_ = list(X_clean.columns)
            self.target_name_ = target_col
            
            # Initialiser le modèle selon le type
            reg_type = self.params['regression_type']
            alpha = self.params['alpha']
            fit_intercept = self.params['fit_intercept']
            
            if reg_type == 'ridge':
                self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
            elif reg_type == 'lasso':
                self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept)
            else:  # linear
                self.model = LinearRegression(fit_intercept=fit_intercept)
                
            # Entraîner le modèle
            self.model.fit(X_clean, y_clean)
            
            # Prédictions
            y_pred = self.model.predict(X_clean)
            
            # Métriques
            r2 = r2_score(y_clean, y_pred)
            mse = mean_squared_error(y_clean, y_pred)
            rmse = np.sqrt(mse)
            
            # Ajouter prédictions au DataFrame
            result_data = data.copy()
            result_data['predicted'] = np.nan
            result_data.loc[mask, 'predicted'] = y_pred
            
            # Résultat
            result = {
                'data': result_data,
                'model': self.model,
                'metrics': {
                    'r2_score': r2,
                    'mse': mse,
                    'rmse': rmse,
                    'n_samples': len(X_clean),
                    'n_features': len(self.feature_names_)
                },
                'coefficients': {
                    'intercept': float(self.model.intercept_) if fit_intercept else 0.0,
                    'features': dict(zip(self.feature_names_, self.model.coef_))
                },
                'info': {
                    'regression_type': reg_type,
                    'alpha': alpha if reg_type in ['ridge', 'lasso'] else None,
                    'feature_names': self.feature_names_,
                    'target_name': self.target_name_
                }
            }
            
            self.logger.info(f"Régression terminée - R² = {r2:.4f}, RMSE = {rmse:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {str(e)}")
            raise


# Auto-enregistrement du bloc
BlockRegistry.register_block('LinearRegressionBlock', LinearRegressionBlock)
