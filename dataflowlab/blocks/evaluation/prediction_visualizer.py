from typing import Any, Dict, Optional
import pandas as pd
import plotly.express as px
from dataflowlab.core.block_base import BlockBase
from dataflowlab.utils.logger import get_logger

class PredictionVisualizerBlock(BlockBase):
    """
    Bloc de visualisation des prédictions vs réalité.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="PredictionVisualizer", params=params)
        self.logger = get_logger("PredictionVisualizer")

    def transform(self, X):
        y_true = self.params.get("y_true")
        y_pred = self.params.get("y_pred")
        fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Vrai", "y": "Prédit"}, title="Prédictions vs Réalité")
        return fig
