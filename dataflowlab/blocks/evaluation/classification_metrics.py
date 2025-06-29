from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class ClassificationMetrics(BlockBase):
    """
    Bloc de calcul des métriques de classification.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="ClassificationMetrics", category="evaluation", params=params)
        self.logger = get_logger("ClassificationMetrics")

    def transform(self, X):
        y_true = self.params.get("y_true")
        y_pred = self.params.get("y_pred")
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if len(set(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            except Exception:
                metrics["roc_auc"] = None
        return metrics
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute classification metrics calculation."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            true_column = self.params.get('true_column', 'y_true')
            pred_column = self.params.get('pred_column', 'y_pred')
            
            if true_column not in data.columns:
                self.logger.error(f"Colonne vraies valeurs '{true_column}' non trouvée")
                return data
            
            if pred_column not in data.columns:
                self.logger.error(f"Colonne prédictions '{pred_column}' non trouvée")
                return data
            
            # Extraire les valeurs vraies et prédites
            mask = data[true_column].notna() & data[pred_column].notna()
            y_true = data.loc[mask, true_column]
            y_pred = data.loc[mask, pred_column]
            
            if len(y_true) == 0:
                self.logger.warning("Aucune donnée valide pour calculer les métriques")
                return data
            
            # Calculer les métriques
            metrics = {}
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Pour la classification binaire, calculer ROC AUC
            unique_classes = len(np.unique(y_true))
            if unique_classes == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except Exception as e:
                    self.logger.warning(f"Impossible de calculer ROC AUC: {str(e)}")
                    metrics['roc_auc'] = None
            
            # Log des métriques
            self.logger.info("Métriques de classification calculées:")
            for metric, value in metrics.items():
                if value is not None:
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            # Ajouter les métriques au DataFrame comme nouvelles colonnes
            result_data = data.copy()
            for metric, value in metrics.items():
                if value is not None:
                    result_data[f'metric_{metric}'] = value
            
            # Ajouter aussi un rapport de classification détaillé
            try:
                report = classification_report(y_true, y_pred, output_dict=True)
                result_data['classification_report'] = str(report)
            except Exception as e:
                self.logger.warning(f"Impossible de générer le rapport de classification: {str(e)}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de classification : {str(e)}")
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
BlockRegistry.register_block('ClassificationMetrics', ClassificationMetrics)
