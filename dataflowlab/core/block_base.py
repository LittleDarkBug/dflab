from typing import Any, Dict, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BlockBase(ABC):
    """
    Classe de base abstraite pour tous les blocs du pipeline DataFlowLab.
    
    Chaque bloc doit implémenter les méthodes execute() et get_config_interface().
    L'architecture supporte à la fois l'interface moderne (execute/validate_inputs/get_config_interface)
    et l'interface legacy (process/get_config_fields) pour compatibilité.
    """
    
    def __init__(self, name: str = None, category: str = "other", 
                 params: Optional[Dict[str, Any]] = None, 
                 default_params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name or self.__class__.__name__
        self.category = category
        self.default_params = default_params or {}
        self.params = {**self.default_params, **(params or {})}
        self._fitted = False
        self._output_data = None
        
        
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Méthode principale d'exécution du bloc.
        
        Args:
            inputs: Dictionnaire des données d'entrée
            
        Returns:
            Dictionnaire contenant les résultats
        """
        # Priorité à la méthode execute() si implémentée
        if hasattr(self, '_execute_impl') and callable(getattr(self, '_execute_impl')):
            return self._execute_impl(inputs)
        
        # Fallback vers process() pour compatibilité legacy
        try:
            # Extraire les données principales
            data = inputs.get('data') 
            if data is None and 'file_path' in inputs:
                # Pour les loaders, utiliser le file_path
                self.params.update(inputs)
                result = self.process()
                return {'data': result, 'info': {'source': 'process_fallback'}}
            elif data is not None:
                result = self.process(data, **inputs)
                return {'data': result, 'info': {'source': 'process_fallback'}}
            else:
                # Cas général - passer tous les inputs
                result = self.process(**inputs)
                return {'data': result, 'info': {'source': 'process_fallback'}}
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du bloc {self.name}: {e}")
            raise
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Valide les données d'entrée (optionnel - implémentation par défaut).
        
        Args:
            inputs: Données d'entrée à valider
            
        Returns:
            Tuple (is_valid, message)
        """
        return True, "OK"
    
    def get_config_interface(self) -> Dict[str, Any]:
        """
        Retourne la configuration de l'interface utilisateur moderne.
        
        Returns:
            Dictionnaire de configuration pour l'UI dynamique
        """
        # Fallback vers get_config_fields() pour compatibilité
        if hasattr(self, 'get_config_fields'):
            try:
                fields = self.get_config_fields()
                config = {}
                for field in fields:
                    field_name = field.get('name', field.get('label', 'unknown'))
                    config[field_name] = {
                        'type': field.get('type', 'text'),
                        'label': field.get('label', field_name),
                        'default': field.get('default', ''),
                        'description': field.get('description', ''),
                        'required': field.get('required', False)
                    }
                    if 'options' in field:
                        config[field_name]['options'] = field['options']
                return config
            except Exception as e:
                logger.warning(f"Erreur lors de la conversion get_config_fields(): {e}")
        
        return {}

    @abstractmethod
    def process(self, data: Union[pd.DataFrame, np.ndarray, Any] = None, **kwargs) -> Any:
        """
        Traite les données d'entrée et retourne le résultat.
        
        Args:
            data: Données d'entrée (optionnel pour certains blocs comme les loaders)
            **kwargs: Arguments supplémentaires
            
        Returns:
            Données transformées
        """
        pass
    
    
    def get_config_fields(self) -> List[Dict[str, Any]]:
        """
        Méthode legacy pour compatibilité.
        Convertit get_config_interface() vers le format legacy.
        """
        try:
            config_interface = self.get_config_interface()
            fields = []
            for name, config in config_interface.items():
                field = {
                    'name': name,
                    'type': config.get('type', 'text'),
                    'label': config.get('label', name),
                    'default': config.get('default', ''),
                    'description': config.get('description', ''),
                    'required': config.get('required', False)
                }
                if 'options' in config:
                    field['options'] = config['options']
                fields.append(field)
            return fields
        except Exception:
            return []
    
    def fit(self, X: Any, y: Any = None) -> 'BlockBase':
        """Ajuste le bloc sur les données d'entraînement."""
        try:
            self._fitted = True
            return self
        except Exception as e:
            logger.error(f"Erreur lors du fit du bloc {self.name}: {e}")
            raise
    
    def transform(self, X: Any) -> Any:
        """Transforme les données d'entrée."""
        try:
            return self.process(X)
        except Exception as e:
            logger.error(f"Erreur lors de la transformation du bloc {self.name}: {e}")
            raise
    
    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Ajuste et transforme les données."""
        return self.fit(X, y).transform(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres du bloc."""
        return self.params.copy()
    
    def set_params(self, **params) -> None:
        """Met à jour les paramètres du bloc."""
        self.params.update(params)
        
    def get_output(self) -> Any:
        """Retourne les dernières données de sortie."""
        return self._output_data
    
    def is_fitted(self) -> bool:
        """Vérifie si le bloc a été ajusté."""
        return self._fitted
    
    def get_info(self) -> Dict[str, Any]:
        """Retourne les informations du bloc."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'params': self.params,
            'fitted': self._fitted
        }
