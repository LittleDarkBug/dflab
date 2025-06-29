from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class OneHotEncoder(BlockBase):
    """
    Bloc d'encodage One-Hot des variables catégorielles avec scikit-learn.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'columns': None,  # Si None, détection automatique des colonnes catégorielles
            'drop': 'first',  # Éviter la multicolinéarité
            'handle_unknown': 'ignore',
            'sparse_output': False
        }
        super().__init__(
            name="OneHotEncoder", 
            category="feature_engineering",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.encoder = None
        self.feature_names_ = None
        self.categorical_columns_ = None

    def get_config_interface(self) -> Dict[str, Any]:
        """Configuration interface pour l'UI."""
        return {
            'columns': {
                'type': 'text',
                'default': '',
                'description': 'Colonnes à encoder (séparées par virgule, vide = auto)'
            },
            'drop': {
                'type': 'select',
                'options': ['first', 'if_binary', None],
                'default': 'first',
                'description': 'Stratégie de suppression pour éviter multicolinéarité'
            },
            'handle_unknown': {
                'type': 'select',
                'options': ['ignore', 'error'],
                'default': 'ignore',
                'description': 'Gestion des valeurs inconnues'
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
            
        return True, "OK"

    def process(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """Implémentation de process pour compatibilité."""
        if data is None:
            data = kwargs.get('data')
        
        result = self._execute_impl({'data': data})
        return result['data']

    def _execute_impl(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute l'encodage One-Hot."""
        try:
            # Validation
            is_valid, msg = self.validate_inputs(inputs)
            if not is_valid:
                raise ValueError(f"Validation échouée: {msg}")
                
            data = inputs['data'].copy()
            
            self.logger.info(f"Démarrage One-Hot Encoding sur {len(data)} échantillons")
            
            # Déterminer les colonnes à encoder
            columns_param = self.params.get('columns')
            if columns_param and columns_param.strip():
                # Colonnes spécifiées par l'utilisateur
                self.categorical_columns_ = [col.strip() for col in columns_param.split(',')]
                # Vérifier que les colonnes existent
                missing_cols = [col for col in self.categorical_columns_ if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"Colonnes introuvables: {missing_cols}")
            else:
                # Détection automatique des colonnes catégorielles
                self.categorical_columns_ = data.select_dtypes(include=['object', 'category']).columns.tolist()
                if not self.categorical_columns_:
                    self.logger.warning("Aucune colonne catégorielle détectée")
                    return {'data': data, 'info': {'message': 'Aucune colonne à encoder'}}
            
            # Initialiser l'encodeur
            self.encoder = SklearnOneHotEncoder(
                drop=self.params['drop'],
                handle_unknown=self.params['handle_unknown'],
                sparse_output=self.params['sparse_output']
            )
            
            # Fit et transform sur les colonnes catégorielles
            encoded_array = self.encoder.fit_transform(data[self.categorical_columns_])
            
            # Créer les noms de colonnes
            self.feature_names_ = self.encoder.get_feature_names_out(self.categorical_columns_)
            
            # Créer DataFrame avec colonnes encodées
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=self.feature_names_,
                index=data.index
            )
            
            # Combiner avec les autres colonnes
            other_columns = [col for col in data.columns if col not in self.categorical_columns_]
            result_data = pd.concat([
                data[other_columns],
                encoded_df
            ], axis=1)
            
            # Résultat
            result = {
                'data': result_data,
                'encoder': self.encoder,
                'info': {
                    'original_columns': self.categorical_columns_,
                    'encoded_columns': self.feature_names_.tolist(),
                    'n_original_features': len(self.categorical_columns_),
                    'n_encoded_features': len(self.feature_names_),
                    'drop_strategy': self.params['drop']
                }
            }
            
            self.logger.info(f"One-Hot Encoding terminé: {len(self.categorical_columns_)} colonnes → {len(self.feature_names_)} features")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {str(e)}")
            raise

# Auto-enregistrement du bloc
BlockRegistry.register_block('OneHotEncoder', OneHotEncoder)
