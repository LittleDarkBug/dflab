class DataFlowLabError(Exception):
    """Exception de base pour DataFlowLab."""
    pass

class BlockExecutionError(DataFlowLabError):
    """Erreur lors de l'exécution d'un bloc."""
    pass
