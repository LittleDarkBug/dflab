from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class DatasetSplitter(BlockBase):
    """
    Bloc de division des données en ensembles train/validation/test avec stratification.
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "DatasetSplitter", params=params, category="data_input")

    def process(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Divise le dataset en ensembles d'entraînement, validation et test.
        
        Args:
            data: DataFrame à diviser
            
        Returns:
            Dictionnaire contenant les différents ensembles
        """
        try:
            if data is None or data.empty:
                raise ValueError("Aucune donnée à diviser")
            
            # Paramètres de division
            test_size = self.params.get("test_size", 0.2)
            val_size = self.params.get("val_size", 0.2)
            target_column = self.params.get("target_column", "")
            stratify = self.params.get("stratify", False)
            random_state = self.params.get("random_state", 42)
            
            # Validation des paramètres
            if test_size + val_size >= 1.0:
                raise ValueError("La somme test_size + val_size doit être < 1.0")
            
            # Préparation des données
            if target_column and target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]
                stratify_data = y if stratify else None
            else:
                X = data
                y = None
                stratify_data = None
            
            # Division initiale train + val / test
            if test_size > 0:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_data
                )
            else:
                X_temp, X_test = X, pd.DataFrame()
                y_temp, y_test = y, pd.Series() if y is not None else None
            
            # Division train / validation
            if val_size > 0 and len(X_temp) > 0:
                val_size_adjusted = val_size / (1 - test_size)
                stratify_temp = y_temp if stratify and y_temp is not None else None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    stratify=stratify_temp
                )
            else:
                X_train, X_val = X_temp, pd.DataFrame()
                y_train, y_val = y_temp, pd.Series() if y_temp is not None else None
            
            # Reconstruction des DataFrames complets
            result = {}
            
            if len(X_train) > 0:
                if y_train is not None:
                    train_df = pd.concat([X_train, y_train], axis=1)
                else:
                    train_df = X_train
                result["train"] = train_df
            
            if len(X_val) > 0:
                if y_val is not None:
                    val_df = pd.concat([X_val, y_val], axis=1)
                else:
                    val_df = X_val
                result["validation"] = val_df
            
            if len(X_test) > 0:
                if y_test is not None:
                    test_df = pd.concat([X_test, y_test], axis=1)
                else:
                    test_df = X_test
                result["test"] = test_df
            
            self._output_data = result
            
            # Log des tailles
            sizes_info = {k: f"{v.shape[0]} lignes" for k, v in result.items()}
            logger.info(f"Dataset divisé: {sizes_info}")
            
            return result
            
        except Exception as e:
            error_msg = f"Erreur lors de la division du dataset: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "test_size",
                "type": "number",
                "label": "Taille ensemble test (0-1)",
                "default": 0.2,
                "min": 0,
                "max": 0.9,
                "step": 0.01
            },
            {
                "name": "val_size",
                "type": "number",
                "label": "Taille ensemble validation (0-1)",
                "default": 0.2,
                "min": 0,
                "max": 0.9,
                "step": 0.01
            },
            {
                "name": "target_column",
                "type": "text",
                "label": "Colonne cible (optionnel)",
                "placeholder": "Nom de la colonne cible pour stratification"
            },
            {
                "name": "stratify",
                "type": "checkbox",
                "label": "Stratification basée sur la cible",
                "default": False
            },
            {
                "name": "random_state",
                "type": "number",
                "label": "Seed aléatoire",
                "default": 42,
                "min": 0
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('DatasetSplitter', DatasetSplitter)
