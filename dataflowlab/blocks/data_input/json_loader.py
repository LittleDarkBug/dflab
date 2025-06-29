from typing import Any, Dict, Optional, List, Union
import pandas as pd
import json
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class JSONLoader(BlockBase):
    """
    Bloc de chargement de fichiers JSON avec parsing automatique.
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "JSONLoader", category="data_input", params=params)

    def process(self, data: Any = None, **kwargs) -> pd.DataFrame:
        """
        Charge un fichier JSON et le convertit en DataFrame.
        
        Returns:
            DataFrame pandas avec les données JSON
        """
        try:
            file_path = self.params.get("file_path", "")
            orient = self.params.get("orient", "records")  # records, index, values, split, table
            lines = self.params.get("lines", False)  # True pour JSON Lines format
            encoding = self.params.get("encoding", "utf-8")
            
            if not file_path:
                raise ValueError("Le chemin du fichier JSON est obligatoire")
            
            if lines:
                # JSON Lines format (un objet JSON par ligne)
                df = pd.read_json(file_path, lines=True, encoding=encoding)
            else:
                # JSON standard
                df = pd.read_json(file_path, orient=orient, encoding=encoding)
            
            self._output_data = df
            logger.info(f"JSON chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            return df
            
        except Exception as e:
            error_msg = f"Erreur lors du chargement JSON: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "file_path",
                "type": "file",
                "label": "Chemin du fichier JSON",
                "required": True,
                "accept": ".json,.jsonl"
            },
            {
                "name": "orient",
                "type": "select",
                "label": "Orientation JSON",
                "default": "records",
                "options": ["records", "index", "values", "split", "table"]
            },
            {
                "name": "lines",
                "type": "checkbox",
                "label": "Format JSON Lines",
                "default": False
            },
            {
                "name": "encoding",
                "type": "select",
                "label": "Encodage",
                "default": "utf-8",
                "options": ["utf-8", "latin-1", "cp1252"]
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('JSONLoader', JSONLoader)
