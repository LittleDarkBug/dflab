"""
Module blocks - Ensemble des blocs disponibles dans DataFlowLab.
"""

# Import des modules de blocs pour auto-découverte
from . import data_input
from . import data_cleaning
from . import feature_engineering
from . import supervised
from . import unsupervised
from . import evaluation
from . import timeseries
from . import advanced

__all__ = [
    'data_input',
    'data_cleaning', 
    'feature_engineering',
    'supervised',
    'unsupervised',
    'evaluation',
    'timeseries',
    'advanced'
]