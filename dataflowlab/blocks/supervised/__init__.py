"""
Module supervised - Blocs d'apprentissage supervisé.
"""

from .linear_regression import LinearRegressionBlock
from .logistic_regression import LogisticRegressionBlock
from .random_forest import RandomForestBlock
from .decision_tree import DecisionTreeBlock
from .knn import KNNBlock
from .gradient_boosting import GradientBoostingBlock
from .naive_bayes import NaiveBayesBlock
from .neural_network import NeuralNetworkBlock

__all__ = [
    'LinearRegressionBlock',
    'LogisticRegressionBlock',
    'RandomForestBlock',
    'DecisionTreeBlock',
    'KNNBlock',
    'GradientBoostingBlock',
    'NaiveBayesBlock',
    'NeuralNetworkBlock'
]