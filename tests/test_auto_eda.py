import pandas as pd
from dataflowlab.eda.auto_eda import AutoEDA

def test_auto_eda_describe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    eda = AutoEDA()
    desc = eda.describe(df)
    assert "describe" in desc
    assert "info" in desc

def test_auto_eda_plot_distributions():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    eda = AutoEDA()
    figs = eda.plot_distributions(df)
    assert isinstance(figs, dict)

def test_auto_eda_detect_outliers():
    df = pd.DataFrame({"a": [1, 100, 3], "b": [4, 5, 100]})
    eda = AutoEDA()
    outliers = eda.detect_outliers(df)
    assert "a" in outliers
    assert "b" in outliers
