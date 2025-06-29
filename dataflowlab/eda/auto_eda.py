import pandas as pd
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
# Import conditionnel de seaborn
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

import plotly.express as px
from dataflowlab.utils.logger import get_logger

logger = get_logger("AutoEDA")

def generate_eda_report(df: pd.DataFrame) -> str:
    """Génère un rapport EDA texte à partir d'un DataFrame."""
    report = []
    
    try:
        # Informations générales
        report.append("# Rapport d'Analyse Exploratoire")
        report.append(f"**Dimensions**: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        # Statistiques descriptives
        desc = df.describe()
        report.append("## Statistiques descriptives")
        report.append(str(desc))
        
        # Valeurs manquantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            report.append("## Valeurs manquantes")
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                report.append(f"- {col}: {count} ({pct:.1f}%)")
        
        # Types de données
        report.append("## Types de données")
        for col, dtype in df.dtypes.items():
            report.append(f"- {col}: {dtype}")
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport EDA : {e}")
        report.append(f"Erreur lors de l'analyse : {e}")
    
    return "\n\n".join(report)


class AutoEDA:
    """
    EDA automatique avec statistiques, visualisations et suggestions.
    """
    def __init__(self) -> None:
        self.logger = get_logger("AutoEDA")

    def describe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Génère des statistiques descriptives."""
        desc = df.describe(include="all").to_dict()
        info = {"shape": df.shape, "dtypes": df.dtypes.astype(str).to_dict()}
        return {"describe": desc, "info": info}

    def plot_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Crée des graphiques de distribution pour chaque colonne."""
        figs = {}
        for col in df.select_dtypes(include=["number", "category", "object"]):
            try:
                figs[col] = px.histogram(df, x=col, title=f"Distribution de {col}")
            except Exception:
                continue
        return figs

    def plot_correlation(self, df: pd.DataFrame) -> Any:
        """Crée une matrice de corrélation."""
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Matrice de corrélation")
        return fig

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Détecte les outliers avec la méthode IQR."""
        outliers = {}
        for col in df.select_dtypes(include=["number"]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
            outliers[col] = df[col][mask].tolist()
        return outliers

    def suggest_preprocessing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Suggère des étapes de prétraitement."""
        suggestions = {}
        if df.isnull().sum().sum() > 0:
            suggestions["missing_values"] = "Imputation recommandée."
        for col in df.select_dtypes(include=["object"]):
            if df[col].nunique() > 20:
                suggestions[col] = "Encodage ou réduction de cardinalité conseillé."
        return suggestions
