from typing import Any, Dict, List
import gradio as gr

class DragDropBlocks:
    """
    Interface drag-and-drop pour la création de pipelines ML.
    """
    def __init__(self, block_types: List[str]) -> None:
        self.block_types = block_types
        self.blocks = []

    def render(self) -> gr.Blocks:
        with gr.Blocks() as demo:
            gr.Markdown("# DataFlowLab - Construction visuelle de pipeline ML")
            with gr.Row():
                block_list = gr.Dropdown(self.block_types, label="Ajouter un bloc", multiselect=False)
                add_btn = gr.Button("Ajouter au pipeline")
            pipeline_view = gr.Markdown("Pipeline courant : (drag & drop à venir)")
        return demo
