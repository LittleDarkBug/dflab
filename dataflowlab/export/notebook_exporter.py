from typing import Any, Dict, List
import nbformat

class NotebookExporter:
    """
    Export Jupyter notebook à partir d'une liste de cellules.
    """
    def export(self, cells: List[Dict[str, Any]], output_path: str) -> None:
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(cell["source"]) if cell["cell_type"] == "code" else nbformat.v4.new_markdown_cell(cell["source"]) for cell in cells]
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

def export_pipeline_to_notebook(pipeline_blocks: List[Dict[str, Any]], output_path: str = "pipeline.ipynb") -> str:
    """
    Exporte un pipeline vers un notebook Jupyter.
    """
    try:
        exporter = NotebookExporter()
        
        # Création des cellules
        cells = [
            {
                "cell_type": "markdown",
                "source": "# Pipeline DataFlowLab\n\nNotebook généré automatiquement"
            },
            {
                "cell_type": "code", 
                "source": "import pandas as pd\nimport numpy as np\nfrom dataflowlab.core.pipeline import Pipeline"
            }
        ]
        
        # Ajout des blocs du pipeline
        for block in pipeline_blocks:
            cells.append({
                "cell_type": "code",
                "source": f"# Bloc: {block.get('type', 'Unknown')}\n# {block.get('params', {})}"
            })
        
        exporter.export(cells, output_path)
        return f"Pipeline exporté vers {output_path}"
        
    except Exception as e:
        return f"Erreur lors de l'export: {e}"
