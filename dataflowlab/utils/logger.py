import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Configure le système de logging global pour DataFlowLab.
    
    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin optionnel vers un fichier de log
        format_string: Format des messages de log
    """
    
    # Configuration du format
    formatter = logging.Formatter(format_string)
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configuration du logger racine
    root_logger = logging.getLogger("dataflowlab")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Handler pour fichier si spécifié
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger configuré pour DataFlowLab.
    
    Args:
        name: Nom du logger (généralement nom de la classe)
        
    Returns:
        Logger configuré
    """
    return logging.getLogger(f"dataflowlab.{name}")

# Configuration par défaut
setup_logging()
