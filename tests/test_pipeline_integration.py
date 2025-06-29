from dataflowlab.core.pipeline import Pipeline
from dataflowlab.blocks.data_input.csv_loader import CSVLoader
from dataflowlab.blocks.data_cleaning.missing_values_handler import MissingValuesHandler
from dataflowlab.blocks.feature_engineering.feature_scaler import FeatureScaler
from dataflowlab.blocks.supervised.random_forest import RandomForestBlock
import pandas as pd
import tempfile
import os

def test_pipeline_integration(tmp_path):
    # Préparer un CSV temporaire
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "y": [0, 1, 0]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    pipeline = Pipeline([
        CSVLoader(params={"path": str(csv_path)}),
        MissingValuesHandler(params={"strategy": "mean"}),
        FeatureScaler(params={"method": "standard"}),
        RandomForestBlock(params={"task": "classification"})
    ])
    X = None
    for block in pipeline.blocks:
        if hasattr(block, "fit_transform"):
            X = block.fit_transform(X, df["y"]) if block.name != "CSVLoader" else block.transform()
        else:
            X = block.transform(X)
    assert X is not None
