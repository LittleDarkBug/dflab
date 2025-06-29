from typing import Any, Dict
import nbformat
from jinja2 import Template

class PipelineExporter:
    """
    Export du pipeline en code Python et Jupyter notebook.
    """
    def export_py(self, code: str, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

    def export_ipynb(self, cells: list, output_path: str) -> None:
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell(cell) for cell in cells]
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    def render_code(self, template_str: str, context: Dict[str, Any]) -> str:
        template = Template(template_str)
        return template.render(**context)
