from typing import Any, Dict, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class TFIDFVectorizerBlock(BlockBase):
    """
    Bloc de vectorisation TF-IDF.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="TFIDFVectorizer", params=params, category="advanced")
        self.logger = get_logger("TFIDFVectorizer")
        self.vectorizer = None

    def fit(self, X, y=None):
        col = self.params.get("col")
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(X[col].astype(str))
        return self

    def transform(self, X):
        col = self.params.get("col")
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer non initialisé. Appelez fit d'abord.")
        arr = self.vectorizer.transform(X[col].astype(str)).toarray()
        columns = self.vectorizer.get_feature_names_out()
        return pd.DataFrame(arr, columns=columns, index=X.index)
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute TF-IDF vectorization on the specified column.
        """
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        text_column = self.params.get("text_column", "text")
        max_features = self.params.get("max_features", 1000)
        ngram_range = self.params.get("ngram_range", (1, 1))
        
        if text_column not in data.columns:
            self.logger.error(f"Colonne '{text_column}' non trouvée dans les données")
            return data
        
        try:
            # Initialize vectorizer with parameters
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
            
            # Fit and transform the text data
            tfidf_matrix = self.vectorizer.fit_transform(data[text_column].astype(str))
            
            # Convert to DataFrame
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"tfidf_{name}" for name in feature_names],
                index=data.index
            )
            
            # Concatenate with original data (excluding text column)
            result_data = pd.concat([
                data.drop(columns=[text_column]),
                tfidf_df
            ], axis=1)
            
            self.logger.info(f"TF-IDF vectorization effectuée avec {len(feature_names)} features")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vectorisation TF-IDF : {str(e)}")
            return data

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Méthode process pour compatibilité avec BlockBase."""
        return self.execute(data, **kwargs)

# Auto-enregistrement du bloc
BlockRegistry.register_block('TFIDFVectorizerBlock', TFIDFVectorizerBlock)
