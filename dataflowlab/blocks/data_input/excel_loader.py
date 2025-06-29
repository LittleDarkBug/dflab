from typing import Any, Dict, Optional, List, Union
import pandas as pd
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class ExcelLoader(BlockBase):
    """
    Bloc de chargement de fichiers Excel avec support multi-feuilles.
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "ExcelLoader", category="data_input", params=params)

    def process(self, data: Any = None, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Charge un fichier Excel avec les paramètres spécifiés.
        
        Returns:
            DataFrame ou dictionnaire de DataFrames (si multi-feuilles)
        """
        try:
            file_path = self.params.get("file_path", "")
            sheet_name = self.params.get("sheet_name", 0)  # 0 = première feuille
            header = self.params.get("header", 0)
            skip_rows = self.params.get("skip_rows", None)
            nrows = self.params.get("nrows", None)
            
            if not file_path:
                raise ValueError("Le chemin du fichier Excel est obligatoire")
            
            # Chargement selon le type de feuille demandé
            if sheet_name == "all":
                # Charger toutes les feuilles
                df_dict = pd.read_excel(
                    file_path,
                    sheet_name=None,
                    header=header,
                    skiprows=skip_rows,
                    nrows=nrows
                )
                self._output_data = df_dict
                total_rows = sum(df.shape[0] for df in df_dict.values())
                logger.info(f"Excel chargé: {len(df_dict)} feuilles, {total_rows} lignes au total")
                return df_dict
            else:
                # Charger une feuille spécifique
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=header,
                    skiprows=skip_rows,
                    nrows=nrows
                )
                self._output_data = df
                logger.info(f"Excel chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                return df
                
        except Exception as e:
            error_msg = f"Erreur lors du chargement Excel: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "file_path",
                "type": "file",
                "label": "Chemin du fichier Excel",
                "required": True,
                "accept": ".xlsx,.xls"
            },
            {
                "name": "sheet_name",
                "type": "text",
                "label": "Nom/Index de la feuille",
                "default": "0",
                "placeholder": "0, 'Sheet1', ou 'all' pour toutes"
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
BlockRegistry.register_block('ExcelLoader', ExcelLoader)
