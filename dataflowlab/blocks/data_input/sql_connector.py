from typing import Any, Dict, Optional, List
import pandas as pd
from sqlalchemy import create_engine, text
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
import logging

logger = logging.getLogger(__name__)

class SQLConnector(BlockBase):
    """
    Bloc de connexion aux bases de données SQLite/PostgreSQL.
    """
    
    def __init__(self, name: str = None, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name or "SQLConnector", category="data_input", params=params)

    def process(self, data: Any = None, **kwargs) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne le résultat sous forme de DataFrame.
        
        Returns:
            DataFrame pandas avec les résultats de la requête
        """
        try:
            # Paramètres de connexion
            db_type = self.params.get("db_type", "sqlite")
            host = self.params.get("host", "localhost")
            port = self.params.get("port", 5432)
            database = self.params.get("database", "")
            username = self.params.get("username", "")
            password = self.params.get("password", "")
            query = self.params.get("query", "")
            
            if not query:
                raise ValueError("La requête SQL est obligatoire")
            
            # Construction de l'URL de connexion
            if db_type == "sqlite":
                if not database:
                    raise ValueError("Le chemin de la base SQLite est obligatoire")
                connection_url = f"sqlite:///{database}"
            elif db_type == "postgresql":
                if not all([host, database, username]):
                    raise ValueError("Host, database et username sont obligatoires pour PostgreSQL")
                connection_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                raise ValueError(f"Type de base de données non supporté: {db_type}")
            
            # Connexion et exécution de la requête
            engine = create_engine(connection_url)
            df = pd.read_sql(text(query), engine)
            
            self._output_data = df
            logger.info(f"Requête SQL exécutée: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            return df
            
        except Exception as e:
            error_msg = f"Erreur lors de l'exécution SQL: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_config_fields(self) -> List[Dict[str, Any]]:
        """Configuration des champs pour l'interface utilisateur."""
        return [
            {
                "name": "db_type",
                "type": "select",
                "label": "Type de base",
                "default": "sqlite",
                "options": ["sqlite", "postgresql"],
                "required": True
            },
            {
                "name": "database",
                "type": "text",
                "label": "Base de données / Fichier SQLite",
                "required": True,
                "placeholder": "chemin/vers/fichier.db ou nom_base"
            },
            {
                "name": "host",
                "type": "text",
                "label": "Hôte (PostgreSQL)",
                "default": "localhost"
            },
            {
                "name": "port",
                "type": "number",
                "label": "Port (PostgreSQL)",
                "default": 5432
            },
            {
                "name": "username",
                "type": "text",
                "label": "Nom d'utilisateur"
            },
            {
                "name": "password",
                "type": "password",
                "label": "Mot de passe"
            },
            {
                "name": "query",
                "type": "textarea",
                "label": "Requête SQL",
                "required": True,
                "placeholder": "SELECT * FROM ma_table WHERE ..."
            }
        ]

# Enregistrement automatique du bloc
BlockRegistry.register_block('SQLConnector', SQLConnector)
