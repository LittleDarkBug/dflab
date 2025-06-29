from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.pipeline import Pipeline

def test_pipeline_fit_transform():
    class DummyBlock(BlockBase):
        def transform(self, X):
            return [x + 1 for x in X]
    blocks = [DummyBlock(name="dummy") for _ in range(2)]
    pipe = Pipeline(blocks)
    X = [1, 2, 3]
    result = pipe.fit_transform(X)
    assert result == [x + 2 for x in X]
