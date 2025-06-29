"""
Bloc Correlation Analysis.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger


class CorrelationAnalysisBlock(BlockBase):
    """
    Bloc d'analyse de corrélation.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params = {
            'method': 'pearson',  # pearson, spearman, kendall
            'threshold': 0.8,  # Seuil pour détecter les corrélations élevées
            'remove_high_corr': False  # Supprimer les features hautement corrélées
        }
        super().__init__(
            name="CorrelationAnalysis",
            category="evaluation",
            params=params,
            default_params=default_params
        )
        self.logger = get_logger(self.__class__.__name__)
        self.correlation_matrix = None
        self.high_corr_pairs = None
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute correlation analysis."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            method = self.params.get('method', 'pearson')
            threshold = self.params.get('threshold', 0.8)
            remove_high_corr = self.params.get('remove_high_corr', False)
            
            # Sélectionner les colonnes numériques
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                self.logger.warning("Aucune colonne numérique trouvée")
                return data
            
            # Calculer la matrice de corrélation
            self.correlation_matrix = numeric_data.corr(method=method)
            
            # Identifier les paires hautement corrélées
            self.high_corr_pairs = []
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    corr_val = abs(self.correlation_matrix.iloc[i, j])
                    if corr_val >= threshold:
                        col1 = self.correlation_matrix.columns[i]
                        col2 = self.correlation_matrix.columns[j]
                        self.high_corr_pairs.append((col1, col2, corr_val))
            
            result_data = data.copy()
            
            # Supprimer les features hautement corrélées si demandé
            if remove_high_corr and self.high_corr_pairs:
                cols_to_remove = set()
                for col1, col2, corr_val in self.high_corr_pairs:
                    # Garder la première colonne, supprimer la seconde
                    cols_to_remove.add(col2)
                
                if cols_to_remove:
                    result_data = result_data.drop(columns=list(cols_to_remove))
                    self.logger.info(f"Colonnes supprimées (corrélation >{threshold}): {list(cols_to_remove)}")
            
            # Ajouter des statistiques de corrélation
            if len(numeric_data.columns) > 1:
                # Corrélation moyenne (en valeur absolue)
                mean_abs_corr = np.abs(self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)]).mean()
                
                # Ajouter au DataFrame
                result_data['corr_mean_abs'] = mean_abs_corr
                result_data['corr_high_pairs_count'] = len(self.high_corr_pairs)
            
            self.logger.info(f"Analyse de corrélation terminée: {len(self.high_corr_pairs)} paires à corrélation élevée (>{threshold})")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur analyse de corrélation : {str(e)}")
            return data



    def process(self, data):
        """Process data using this block."""
        try:
            self.logger.info(f"Traitement des données avec {self.__class__.__name__}")
            
            # Si le bloc a une méthode transform, l'utiliser
            if hasattr(self, 'transform') and callable(getattr(self, 'transform')):
                return self.transform(data)
            
            # Si le bloc a une méthode fit_transform, l'utiliser
            elif hasattr(self, 'fit_transform') and callable(getattr(self, 'fit_transform')):
                return self.fit_transform(data)
            
            # Si le bloc a fit et transform séparément
            elif hasattr(self, 'fit') and hasattr(self, 'transform'):
                self.fit(data)
                return self.transform(data)
            
            # Sinon, retourner les données telles quelles
            else:
                self.logger.warning(f"{self.__class__.__name__} n'a pas de méthode de traitement définie")
                return data
                
        except Exception as e:
            self.logger.error(f"Erreur dans {self.__class__.__name__}: {str(e)}")
            # En cas d'erreur, retourner les données originales
            return data


# Auto-enregistrement du bloc
BlockRegistry.register_block('CorrelationAnalysisBlock', CorrelationAnalysisBlock)
