import pytest
from dataflowlab.core.block_base import BlockBase

def test_block_base_init():
    block = BlockBase(name="test", params={"a": 1})
    assert block.name == "test"
    assert block.params["a"] == 1

def test_block_base_fit_transform():
    block = BlockBase(name="test")
    X = [1, 2, 3]
    assert block.fit(X) is block
    assert block.transform(X) == X
    assert block.fit_transform(X) == X

def test_block_base_params():
    block = BlockBase(name="test", params={"a": 1})
    block.set_params(b=2)
    params = block.get_params()
    assert params["a"] == 1
    assert params["b"] == 2
