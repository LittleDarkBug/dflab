import nbformat
from dataflowlab.export.notebook_exporter import NotebookExporter
import tempfile
import os

def test_notebook_exporter():
    exporter = NotebookExporter()
    cells = [
        {"cell_type": "markdown", "source": "# Titre"},
        {"cell_type": "code", "source": "print('Hello')"}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.ipynb")
        exporter.export(cells, out_path)
        nb = nbformat.read(out_path, as_version=4)
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "code"
        assert "Hello" in nb.cells[1].source
