from typing import Any, Dict
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import os

class ReportExporter:
    """
    Export du rapport EDA au format HTML.
    """
    def __init__(self, template_dir: str = "templates") -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def export_html(self, context: Dict[str, Any], output_path: str) -> None:
        template = self.env.get_template("eda_report.html")
        html = template.render(**context)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
