# DataFlowLab


# Plateforme Visuelle de Pipelines Machine Learning

DataFlowLab est une application Python locale et modulaire permettant aux data scientists de créer visuellement des pipelines ML par drag-and-drop, avec EDA automatisée, génération de code et assistance par LLM local.

## Fonctionnalités

- Construction visuelle de pipelines : Interface drag-and-drop intuitive
- EDA automatique : Analyse exploratoire intelligente avec insights
- Assistant LLM local : Aide contextuelle sans dépendance externe
- Génération de code : Export Python/Jupyter exécutable
- Plus de 40 blocs ML : Bibliothèque complète pour tous types de projets
- Sauvegarde/chargement : Projets persistants et réutilisables

## Installation

### Prérequis

- Python 3.11+
- Gestionnaire de packages pip

### Installation rapide


1. Cloner le projet :

```bash
git clone <repo-url>
cd dataflowlab
```


2. Créer un environnement virtuel :

```bash
# Avec conda (recommandé)
conda create -n dataflowlab python=3.11
conda activate dataflowlab

# Ou avec venv
python -m venv dataflowlab
# Windows
dataflowlab\Scripts\activate
# Linux/Mac
source dataflowlab/bin/activate
```


3. Installer les dépendances :

```bash
pip install -r requirements.txt
```



4. Vérifier l'installation :

Il n'y a plus de script de test rapide universel. Pour vérifier l'installation, lancez simplement l'application ou exécutez les tests unitaires :

```bash
python -m dataflowlab.ui.app
# ou
pytest tests/
```


## Lancement de l'application

```bash
python -m dataflowlab.ui.app
```

L'interface Gradio sera accessible à : `http://localhost:7860`

## Utilisation



### 1. Construction et exécution d’un pipeline

#### Workflow utilisateur

1. Sélection d’un bloc : Cliquez sur un bloc dans la bibliothèque (aucun ajout automatique).
2. Configuration : Les paramètres du bloc s’affichent dynamiquement dans le panneau latéral (ex : fichier à charger, séparateur, options avancées).
3. Ajout explicite : Cliquez sur « Ajouter au pipeline » pour insérer le bloc dans le workflow.
4. Réorganisation : Glissez-déposez pour réordonner les blocs si besoin.
5. Exécution : Cliquez sur « Exécuter Pipeline » pour lancer le traitement séquentiel (chaque bloc applique sa méthode `process`).
6. Visualisation : Les résultats intermédiaires et finaux s’affichent, les erreurs sont expliquées clairement.

#### Conseils

- Aucun bloc n’est ajouté sans action explicite.
- Les paramètres sont adaptés à chaque bloc (data_input, modèles, etc.).
- Les erreurs de type, de format ou de dépendance sont gérées et affichées à l’utilisateur.

#### Schéma du workflow (texte)

Sélection bloc → Configuration → Ajout pipeline → Réorganisation → Exécution → Résultats/Export

### 2. Types de blocs disponibles

Voir la section détaillée plus bas pour la liste complète (Data Input, Cleaning, Feature Engineering, Supervised, Unsupervised, Evaluation, Timeseries, Advanced).

### 3. Gestion avancée des paramètres

- Chaque bloc propose des champs de configuration dynamiques (texte, nombre, select, fichier, etc.).
- Les blocs data_input permettent la sélection de fichiers, le choix du séparateur, l’encodage, etc.
- Les blocs avancés proposent des options expertes accessibles.

### 4. Gestion des erreurs et robustesse

- Les erreurs de configuration, de format de fichier ou de dépendance sont interceptées et expliquées.
- Les messages d’interface guident l’utilisateur à chaque étape.

### 5. Export, sauvegarde et assistant LLM

- Export Python/Jupyter, sauvegarde/chargement JSON, assistant LLM local pour l’aide contextuelle.

### 6. FAQ

Q : Pourquoi mon bloc n’apparaît pas ?
A : Vérifiez qu’il hérite bien de `BlockBase`, qu’il est enregistré dans le `BlockRegistry` et qu’il implémente la méthode `process`.

Q : J’ai une erreur “dépendance manquante” ?
A : Installez le package requis (voir la section Dépendances). Certains blocs avancés nécessitent `statsmodels`, `shap`, etc.

Q : Comment corriger une erreur de type ou de format ?
A : Vérifiez la configuration du bloc, le type de fichier ou le format des données d’entrée.


## Structure du projet


```
dataflowlab/
├── core/                 # Système central
│   ├── block_base.py
│   ├── block_registry.py
│   └── pipeline.py
├── blocks/               # Bibliothèque de blocs ML
│   ├── advanced/
│   ├── data_cleaning/
│   ├── data_input/
│   ├── evaluation/
│   ├── feature_engineering/
│   ├── supervised/
│   ├── timeseries/
│   └── unsupervised/
├── eda/
│   ├── auto_eda.py
│   └── report_exporter.py
├── export/
│   ├── exporter.py
│   └── notebook_exporter.py
├── llm/
│   └── assistant.py
├── ui/
│   ├── app.py
│   └── dragdrop_blocks.py
├── utils/
│   ├── errors.py
│   └── logger.py
└── __init__.py
tests/
├── test_association_rules.py
├── test_auto_eda.py
├── test_block_base.py
├── test_csv_loader.py
├── test_dragdrop_blocks.py
├── test_llm_assistant.py
├── test_notebook_exporter.py
├── test_pipeline.py
└── test_pipeline_integration.py
sample_data.csv
requirements.txt
README.md
DEVELOPER_GUIDE.md
```

## Développement

### Ajouter un nouveau bloc

1. **Créer la classe** dans le module approprié :
```python
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry

class MonNouveauBloc(BlockBase):
    def process(self, data, **kwargs):
        # Votre logique ici
        return data_transformee
    
    def get_config_fields(self):
        return [
            {
                "name": "param1",
                "type": "number",
                "label": "Mon paramètre",
                "default": 0
            }
        ]

# Enregistrement automatique
BlockRegistry.register("MonNouveauBloc", MonNouveauBloc, "ma_categorie")
```

2. **Ajouter aux imports** dans `__init__.py` du module


3. **Tester** avec `pytest tests/`

### Tests

```bash
# Lancer tous les tests unitaires
pytest tests/
```

## Dépendances

### Obligatoires
- `gradio>=4.0` : Interface utilisateur
- `pandas>=2.0` : Manipulation données
- `numpy>=1.24` : Calcul numérique
- `scikit-learn>=1.3` : Machine learning
- `plotly>=5.0` : Visualisations

### Optionnelles
- `statsmodels>=0.14` : Statistiques avancées
- `xgboost>=2.0` : Gradient boosting
- `lightgbm>=4.0` : Gradient boosting
- `llama-cpp-python>=0.2.72` : LLM local
- `shap>=0.44` : Explainability

## Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Créer une Pull Request

## Licence

Ce projet est distribué sous licence GPLv3. Voir le fichier `LICENSE` pour plus de détails.

## Support

- 📚 Documentation complète : `DEVELOPER_GUIDE.md`
- 🐛 Issues : GitHub Issues
- 💬 Discussions : GitHub Discussions

---

**DataFlowLab** - Démocratiser le Machine Learning par la simplicité visuelle.
