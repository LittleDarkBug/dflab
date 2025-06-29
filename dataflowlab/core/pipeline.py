from typing import List, Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from .block_base import BlockBase
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Pipeline modulaire avancé pour l'exécution séquentielle de blocs ML.
    Inclut validation, logging, sauvegarde d'état et gestion d'erreurs.
    """
    
    def __init__(self, blocks: List[BlockBase] = None, name: str = None) -> None:
        self.blocks = blocks or []
        self.name = name or f"Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._execution_log = []
        self._intermediate_results = {}
        self._fitted = False

    def add_block(self, block: BlockBase) -> None:
        """Ajoute un bloc au pipeline."""
        if not isinstance(block, BlockBase):
            raise TypeError("Le bloc doit hériter de BlockBase")
        self.blocks.append(block)
        logger.info(f"Bloc {block.name} ajouté au pipeline {self.name}")

    def remove_block(self, index: int) -> None:
        """Supprime un bloc du pipeline."""
        if 0 <= index < len(self.blocks):
            removed_block = self.blocks.pop(index)
            logger.info(f"Bloc {removed_block.name} supprimé du pipeline")
        else:
            raise IndexError(f"Index {index} hors limites")

    def move_block(self, from_index: int, to_index: int) -> None:
        """Déplace un bloc dans le pipeline."""
        if 0 <= from_index < len(self.blocks) and 0 <= to_index < len(self.blocks):
            block = self.blocks.pop(from_index)
            self.blocks.insert(to_index, block)
            logger.info(f"Bloc déplacé de {from_index} vers {to_index}")

    def fit(self, X: Any, y: Any = None) -> 'Pipeline':
        """Ajuste tous les blocs du pipeline."""
        try:
            current_X = X
            current_y = y
            
            for i, block in enumerate(self.blocks):
                logger.info(f"Ajustement du bloc {i+1}/{len(self.blocks)}: {block.name}")
                
                # Sauvegarde de l'état intermédiaire
                self._intermediate_results[f"step_{i}_input"] = current_X
                
                # Ajustement du bloc
                block.fit(current_X, current_y)
                
                # Transformation pour le bloc suivant
                if hasattr(block, 'transform'):
                    current_X = block.transform(current_X)
                    self._intermediate_results[f"step_{i}_output"] = current_X
                
                # Log de l'exécution
                self._execution_log.append({
                    'step': i,
                    'block_name': block.name,
                    'block_type': block.__class__.__name__,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
            
            self._fitted = True
            return self
            
        except Exception as e:
            error_msg = f"Erreur lors de l'ajustement du pipeline: {e}"
            logger.error(error_msg)
            self._execution_log.append({
                'step': i,
                'block_name': block.name if 'block' in locals() else 'unknown',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise

    def transform(self, X: Any) -> Any:
        """Transforme les données avec tous les blocs."""
        if not self._fitted:
            logger.warning("Pipeline non ajusté. Utilisation de fit_transform recommandée.")
        
        try:
            current_X = X
            
            for i, block in enumerate(self.blocks):
                logger.debug(f"Transformation bloc {i+1}/{len(self.blocks)}: {block.name}")
                current_X = block.transform(current_X)
            
            return current_X
            
        except Exception as e:
            error_msg = f"Erreur lors de la transformation: {e}"
            logger.error(error_msg)
            raise

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Ajuste et transforme les données."""
        self.fit(X, y)
        return self.transform(X)

    def get_execution_log(self) -> List[Dict]:
        """Retourne le log d'exécution."""
        return self._execution_log.copy()

    def get_intermediate_result(self, step: int) -> Any:
        """Récupère un résultat intermédiaire."""
        key = f"step_{step}_output"
        return self._intermediate_results.get(key)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Retourne les informations du pipeline."""
        return {
            'name': self.name,
            'num_blocks': len(self.blocks),
            'blocks': [block.get_info() for block in self.blocks],
            'fitted': self._fitted,
            'execution_log': self._execution_log
        }

    def save_to_file(self, filepath: str) -> None:
        """Sauvegarde la configuration du pipeline."""
        config = {
            'name': self.name,
            'blocks': [
                {
                    'type': block.__class__.__name__,
                    'name': block.name,
                    'params': block.get_params()
                }
                for block in self.blocks
            ],
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline sauvegardé dans {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Pipeline':
        """Charge un pipeline depuis un fichier."""
        from .block_registry import BlockRegistry
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        pipeline = cls(name=config.get('name', 'Loaded_Pipeline'))
        
        for block_config in config['blocks']:
            block_cls = BlockRegistry.get_block(block_config['type'])
            block = block_cls(
                name=block_config['name'],
                params=block_config['params']
            )
            pipeline.add_block(block)
        
        logger.info(f"Pipeline chargé depuis {filepath}")
        return pipeline

    def validate(self) -> List[str]:
        """Valide la configuration du pipeline."""
        errors = []
        
        if not self.blocks:
            errors.append("Pipeline vide")
        
        for i, block in enumerate(self.blocks):
            try:
                # Vérification basique de la configuration
                config_fields = block.get_config_fields()
                if not config_fields:
                    errors.append(f"Bloc {i+1} ({block.name}): pas de configuration")
            except Exception as e:
                errors.append(f"Bloc {i+1} ({block.name}): erreur de validation - {e}")
        
        return errors
