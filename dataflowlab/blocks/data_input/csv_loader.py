from typing import Any, Dict, Optional, List, Union
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class CSVLoader(BlockBase):
    """
    Bloc de chargement de fichiers CSV avec options avancées d'encodage et de parsing.
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "CSVLoader", category="data_input", params=params)

    def process(self, data: Any = None, **kwargs) -> pd.DataFrame:
        """
        Charge un fichier CSV avec les paramètres spécifiés.
        
        Returns:
            DataFrame pandas avec les données chargées
        """
        try:
            # Paramètres avec valeurs par défaut
            file_path = self.params.get("file_path", "")
            encoding = self.params.get("encoding", "utf-8")
            separator = self.params.get("separator", ",")
            decimal = self.params.get("decimal", ".")
            header = self.params.get("header", 0)
            skip_rows = self.params.get("skip_rows", None)
            nrows = self.params.get("nrows", None)
            na_values = self.params.get("na_values", None)
            
            if not file_path:
                raise ValueError("Le chemin du fichier CSV est obligatoire")
            
            # Chargement du CSV
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=separator,
                decimal=decimal,
                header=header,
                skiprows=skip_rows,
                nrows=nrows,
                na_values=na_values
            )
            
            self._output_data = df
            logger.info(f"CSV chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            return df
            
        except Exception as e:
            error_msg = f"Erreur lors du chargement CSV: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "file_path",
                "type": "file",
                "label": "Chemin du fichier CSV",
                "required": True,
                "accept": ".csv"
            },
            {
                "name": "encoding", 
                "type": "select",
                "label": "Encodage",
                "default": "utf-8",
                "options": ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
            },
            {
                "name": "separator",
                "type": "text",
                "label": "Séparateur",
                "default": ",",
                "placeholder": "Virgule, point-virgule, tabulation..."
            },
            {
                "name": "decimal",
                "type": "text", 
                "label": "Séparateur décimal",
                "default": ".",
                "placeholder": ". ou ,"
            },
            {
                "name": "header",
                "type": "number",
                "label": "Ligne d'en-tête",
                "default": 0,
                "min": 0
            },
            {
                "name": "skip_rows",
                "type": "number",
                "label": "Lignes à ignorer (optionnel)",
                "min": 0
            },
            {
                "name": "nrows",
                "type": "number", 
                "label": "Nombre max de lignes (optionnel)",
                "min": 1
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('CSVLoader', CSVLoader)
