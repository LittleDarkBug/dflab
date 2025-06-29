from typing import Any, Dict, Optional, List
import pandas as pd
import json
from pathlib import Path
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class DataExporter(BlockBase):
    """
    Bloc d'exportation de données vers différents formats (CSV, Excel, JSON).
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "DataExporter", category="data_input", params=params)

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Exporte les données vers le format spécifié et retourne les données inchangées.
        
        Args:
            data: DataFrame à exporter
            
        Returns:
            DataFrame inchangé (passthrough)
        """
        try:
            if data is None or data.empty:
                raise ValueError("Aucune donnée à exporter")
            
            file_path = self.params.get("file_path", "")
            file_format = self.params.get("format", "csv")
            
            if not file_path:
                raise ValueError("Le chemin de fichier de sortie est obligatoire")
            
            # Créer le répertoire si nécessaire
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export selon le format
            if file_format == "csv":
                encoding = self.params.get("encoding", "utf-8")
                separator = self.params.get("separator", ",")
                index = self.params.get("include_index", False)
                
                data.to_csv(
                    file_path,
                    encoding=encoding,
                    sep=separator,
                    index=index
                )
                
            elif file_format == "excel":
                sheet_name = self.params.get("sheet_name", "Sheet1")
                index = self.params.get("include_index", False)
                
                data.to_excel(
                    file_path,
                    sheet_name=sheet_name,
                    index=index
                )
                
            elif file_format == "json":
                orient = self.params.get("orient", "records")
                encoding = self.params.get("encoding", "utf-8")
                
                data.to_json(
                    file_path,
                    orient=orient,
                    force_ascii=False
                )
                
            else:
                raise ValueError(f"Format non supporté: {file_format}")
            
            self._output_data = data
            logger.info(f"Données exportées vers {file_path} (format: {file_format})")
            
            return data
            
        except Exception as e:
            error_msg = f"Erreur lors de l'export: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "file_path",
                "type": "text",
                "label": "Chemin de fichier de sortie",
                "required": True,
                "placeholder": "chemin/vers/fichier.csv"
            },
            {
                "name": "format",
                "type": "select",
                "label": "Format de sortie",
                "default": "csv",
                "options": ["csv", "excel", "json"],
                "required": True
            },
            {
                "name": "encoding",
                "type": "select",
                "label": "Encodage (CSV/JSON)",
                "default": "utf-8",
                "options": ["utf-8", "latin-1", "cp1252"]
            },
            {
                "name": "separator",
                "type": "text",
                "label": "Séparateur (CSV)",
                "default": ",",
                "placeholder": "Virgule, point-virgule..."
            },
            {
                "name": "sheet_name",
                "type": "text",
                "label": "Nom de feuille (Excel)",
                "default": "Sheet1"
            },
            {
                "name": "orient",
                "type": "select",
                "label": "Orientation (JSON)",
                "default": "records",
                "options": ["records", "index", "values", "split", "table"]
            },
            {
                "name": "include_index",
                "type": "checkbox",
                "label": "Inclure l'index",
                "default": False
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('DataExporter', DataExporter)
