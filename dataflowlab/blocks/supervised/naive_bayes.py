from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.utils.logger import get_logger

class NaiveBayesBlock(BlockBase):
    """
    Bloc Naive Bayes (GaussianNB, MultinomialNB).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name="NaiveBayes", params=params)
        self.logger = get_logger("NaiveBayes")
        self.model = None

    def fit(self, X, y):
        algo = self.params.get("algo", "gaussian")
        if algo == "multinomial":
            self.model = MultinomialNB()
        else:
            self.model = GaussianNB()
        self.model.fit(X, y)
        return self

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez fit d'abord.")
        return pd.Series(self.model.predict(X), index=X.index, name="prediction")
    
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute Naive Bayes training and prediction."""
        if data is None or data.empty:
            self.logger.warning("Données vides fournies")
            return pd.DataFrame()
            
        try:
            target_column = self.params.get("target_column", "target")
            algo = self.params.get("algo", "gaussian")
            
            if target_column not in data.columns:
                self.logger.error(f"Colonne cible '{target_column}' non trouvée")
                return data
                
            # Préparer les données
            y = data[target_column]
            X = data.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
            
            # Supprimer les lignes avec des valeurs manquantes
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                self.logger.warning("Aucune donnée valide après nettoyage")
                return data
                
            # Initialiser le modèle selon l'algorithme
            if algo == "multinomial":
                # Pour MultinomialNB, les données doivent être non-négatives
                X_clean = X_clean.abs()
                self.model = MultinomialNB()
            else:
                self.model = GaussianNB()
                
            # Entraîner le modèle
            self.model.fit(X_clean, y_clean)
            
            # Prédictions
            y_pred = self.model.predict(X_clean)
            
            # Ajouter prédictions au DataFrame
            result_data = data.copy()
            result_data['nb_prediction'] = np.nan
            result_data.loc[mask, 'nb_prediction'] = y_pred
            
            # Calculer les métriques (Naive Bayes est pour la classification)
            accuracy = accuracy_score(y_clean, y_pred)
            self.logger.info(f"Naive Bayes accuracy: {accuracy:.4f}")
            
            self.logger.info(f"Naive Bayes ({algo}) entraîné avec {len(X_clean)} échantillons")
            return result_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de Naive Bayes : {str(e)}")
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
BlockRegistry.register_block('NaiveBayesBlock', NaiveBayesBlock)
