from typing import Any, Dict, Optional
import pandas as pd
from PIL import Image
import os
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class ImageLoader(BlockBase):
    """
    Bloc de chargement et préprocessing d'images.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="ImageLoader", params=params, category="advanced")
        self.logger = get_logger("ImageLoader")

    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute image loading."""
        try:
            return self.transform(data)
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement d'images : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Méthode process pour compatibilité avec BlockBase."""
        return self.execute(data, **kwargs)

    def transform(self, X=None):
        folder = self.params.get("folder")
        size = self.params.get("size", (224, 224))
        images = []
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(os.path.join(folder, fname)).resize(size)
                images.append(img)
        return images

# Auto-enregistrement du bloc
BlockRegistry.register_block('ImageLoader', ImageLoader)
