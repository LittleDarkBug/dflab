import pandas as pd
import pytest
from dataflowlab.blocks.data_input.csv_loader import CSVLoader

def test_csv_loader(tmp_path):
    # Création d'un fichier CSV temporaire
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    loader = CSVLoader(params={"path": str(csv_path)})
    loaded = loader.transform()
    assert loaded.equals(df)
