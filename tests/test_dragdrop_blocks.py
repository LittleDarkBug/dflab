import gradio as gr
from dataflowlab.ui.dragdrop_blocks import DragDropBlocks

def test_dragdrop_blocks_render():
    block_types = ["CSVLoader", "RandomForestBlock"]
    ui = DragDropBlocks(block_types)
    demo = ui.render()
    assert isinstance(demo, gr.Blocks)
