import pandas as pd
from dataflowlab.blocks.unsupervised.association_rules import AssociationRulesBlock

def test_association_rules_block():
    # Exemple de transactions (one-hot encoded)
    df = pd.DataFrame({
        'milk': [1, 0, 1, 1],
        'bread': [1, 1, 0, 1],
        'butter': [0, 1, 1, 1]
    })
    block = AssociationRulesBlock(params={"min_support": 0.5, "min_confidence": 0.5})
    rules = block.transform(df)
    assert not rules.empty
    assert 'confidence' in rules.columns
