
# Guide Développeur — DataFlowLab

## Architecture du Projet

DataFlowLab est conçu selon une architecture modulaire et extensible permettant l'ajout facile de nouveaux blocs et fonctionnalités.

### Composants Principaux

#### 1. Core System (`dataflowlab/core/`)

**BlockBase** (`block_base.py`)
- Classe abstraite de base pour tous les blocs
- Définit l'interface standard : `process()`, `get_config_fields()`
- Gestion automatique du fit/transform et de l'état

**BlockRegistry** (`block_registry.py`)
- Registre centralisé des blocs disponibles
- Auto-découverte des blocs par import
- Organisation par catégories

**Pipeline** (`pipeline.py`)
- Orchestration de l'exécution séquentielle des blocs
- Gestion des états intermédiaires et logging
- Sauvegarde/chargement de configuration

#### 2. Block Library (`dataflowlab/blocks/`)

Bibliothèque de 40+ blocs organisés par catégories :

- **data_input** : Chargement données (CSV, Excel, JSON, SQL)
- **data_cleaning** : Nettoyage (valeurs manquantes, outliers, doublons)
- **feature_engineering** : Transformation features (scaling, encoding, sélection)
- **supervised** : Modèles supervisés (régression, classification)
- **unsupervised** : Clustering, réduction dimensionnalité
- **evaluation** : Métriques, validation croisée, explainability
- **timeseries** : Séries temporelles spécialisées
- **advanced** : Traitements avancés (NLP, images, ensembles)

#### 3. User Interface (`dataflowlab/ui/`)

**Interface Gradio** (`app.py`)
- Interface web responsive avec drag-and-drop
- Configuration dynamique des blocs
- Visualisation temps réel du pipeline
- Export de code et projets

#### 4. Supporting Modules

- **eda/** : Analyse exploratoire automatique
- **export/** : Génération code Python/Jupyter
- **llm/** : Assistant LLM local
- **utils/** : Logging, gestion erreurs

## Développement de Nouveaux Blocs


### Structure standard d’un bloc

Chaque bloc doit :
- Hériter de `BlockBase`
- Implémenter obligatoirement la méthode `process(self, data, **kwargs)`
- S’auto-enregistrer dans le `BlockRegistry` (catégorie obligatoire)
- Définir les champs de configuration dynamiques via `get_config_fields()`

Exemple minimal :

```python
from dataflowlab.core.block_base import BlockBase
from dataflowlab.core.block_registry import BlockRegistry

class MonBloc(BlockBase):
    def __init__(self, params=None):
        super().__init__(name="MonBloc", params=params, category="ma_categorie")

    def process(self, data, **kwargs):
        # Traitement principal
        return data

    def get_config_fields(self):
        return [
            {"name": "param1", "type": "number", "label": "Mon paramètre", "default": 0}
        ]

BlockRegistry.register_block('MonBloc', MonBloc)
```


### Types de champs de configuration

- text : Champ texte libre
- number : Champ numérique avec min/max/step
- select : Liste déroulante avec options
- checkbox : Case à cocher booléenne
- textarea : Zone de texte multi-lignes
- file : Sélecteur de fichier avec filtres
- password : Champ mot de passe masqué

## Installation et Développement

### Prérequis
- Python 3.11+
- pip pour l'installation des packages


### Mise en place de l’environnement

```bash
git clone <repo>
cd dataflowlab
conda create -n dataflowlab python=3.11
conda activate dataflowlab
pip install -r requirements.txt
python simple_test.py
```


### Lancement pour développement

```bash
python -m dataflowlab.ui.app
python test_dataflowlab.py
```

## Conventions et Standards

### Code Style
- PascalCase pour les classes
- snake_case pour fonctions/variables
- Type hints obligatoires
- Docstrings pour méthodes publiques
- Gestion d'erreurs avec logging


### Architecture des blocs

1. Hériter de `BlockBase`
2. Implémenter `process()` et `get_config_fields()`
3. Enregistrer avec `BlockRegistry.register_block()`
4. Ajouter aux imports du module


### Tests et validation

- Tests unitaires pour chaque bloc
- Validation des paramètres d'entrée
- Gestion robuste des erreurs
- Logging approprié des opérations


## Contribution

1. Fork du projet
2. Création d’une branche feature
3. Développement avec tests
4. Pull Request

---

Pour des détails spécifiques, consultez le code source ou ouvrez une issue GitHub.
