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


## Paramètres de configuration des blocs (JSON)

### Bloc CSVLoader

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| file_path     | file     | .csv                                | Chemin du fichier CSV à charger                  |
| encoding      | select   | utf-8, latin-1, cp1252, iso-8859-1  | Encodage du fichier                             |
| separator     | text     | , ; \t                             | Séparateur de colonnes (virgule, point-virgule…) |
| decimal       | text     | . ou ,                              | Séparateur décimal                              |
| header        | number   | 0, 1, ...                           | Ligne d'en-tête (0 = première ligne)             |
| skip_rows     | number   | 0, 1, ...                           | Nombre de lignes à ignorer au début (optionnel)  |
| nrows         | number   | 10, 1000, ...                       | Nombre max de lignes à lire (optionnel)          |

### Bloc ExcelLoader

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| file_path     | file     | .xlsx, .xls                         | Chemin du fichier Excel à charger                |
| sheet_name    | text     | 0, 'Sheet1', 'all'                  | Feuille à charger (index, nom ou 'all')          |
| header        | number   | 0, 1, ...                           | Ligne d'en-tête (0 = première ligne)             |
| skip_rows     | number   | 0, 1, ...                           | Lignes à ignorer au début (optionnel)            |
| nrows         | number   | 10, 1000, ...                       | Nombre max de lignes à lire (optionnel)          |

### Bloc JSONLoader

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| file_path     | file     | .json, .jsonl                       | Chemin du fichier JSON à charger                 |
| orient        | select   | records, index, values, split, table| Orientation du JSON pour pandas                  |
| lines         | checkbox | true / false                        | Format JSON Lines (un objet par ligne)           |
| encoding      | select   | utf-8, latin-1, cp1252              | Encodage du fichier                             |

### Bloc SQLConnector

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| db_type       | select   | sqlite, postgresql                  | Type de base de données                          |
| database      | text     | chemin/vers/fichier.db, nom_base    | Fichier SQLite ou nom de la base PostgreSQL      |
| host          | text     | localhost, 127.0.0.1                | Hôte de la base (PostgreSQL)                     |
| port          | number   | 5432, ...                           | Port de connexion (PostgreSQL)                   |
| username      | text     | utilisateur                         | Nom d'utilisateur (PostgreSQL)                   |
| password      | password | *****                               | Mot de passe (PostgreSQL)                        |
| query         | textarea | SELECT * FROM table                  | Requête SQL à exécuter                           |

### Bloc MissingValuesHandler

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| strategy      | select   | mean, median, mode, constant, knn, iterative | Stratégie d’imputation des valeurs manquantes    |
| fill_value    | number   | 0, 1, ...                           | Valeur utilisée si strategy=constant             |
| n_neighbors   | number   | 1-20                                 | Nombre de voisins pour la méthode KNN            |
| max_iter      | number   | 1-100                                | Nombre d’itérations pour la méthode iterative    |

### Bloc FeatureScaler

| Paramètre          | Type     | Valeurs possibles / Exemple         | Description                                      |
|--------------------|----------|-------------------------------------|--------------------------------------------------|
| scaler_type        | select   | standard, minmax, robust, maxabs    | Type de normalisation/scaling à appliquer        |
| feature_range_min  | number   | 0, -1, ...                          | Borne min (pour MinMaxScaler)                   |
| feature_range_max  | number   | 1, 10, ...                          | Borne max (pour MinMaxScaler)                   |

### Bloc DuplicateRemover

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| subset        | text     | col1,col2,... / None                | Colonnes à considérer pour la déduplication (None = toutes) |
| keep          | select   | first, last, False                  | Quelle occurrence conserver (première, dernière, aucune)     |
| ignore_index  | checkbox | true / false                        | Réindexer le DataFrame après suppression         |

### Bloc OneHotEncoder

| Paramètre        | Type     | Valeurs possibles / Exemple         | Description                                      |
|------------------|----------|-------------------------------------|--------------------------------------------------|
| columns          | text     | col1,col2,... / vide                | Colonnes à encoder (vide = auto-détection)       |
| drop             | select   | first, if_binary, None              | Stratégie de suppression pour multicolinéarité   |
| handle_unknown   | select   | ignore, error                       | Gestion des valeurs inconnues                    |
| sparse_output    | checkbox | true / false                        | Format de sortie sparse (avancé)                 |

### Bloc LabelEncoder

| Paramètre            | Type     | Valeurs possibles / Exemple         | Description                                      |
|----------------------|----------|-------------------------------------|--------------------------------------------------|
| categorical_columns  | text     | col1,col2,... / vide                | Colonnes à encoder (vide = auto-détection)       |

> Pour chaque bloc, la configuration JSON doit respecter ces paramètres pour garantir un fonctionnement correct dans le pipeline.

### Bloc BinningTransformer

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| n_bins      | number | 2, 5, 10                        | Nombre de bins/discrétisations à créer           |
| strategy    | select | uniform, quantile, kmeans        | Stratégie de discrétisation utilisée             |

### Bloc DateFeatureExtractor

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| (aucun)     |        |                                  | Ce bloc n’a pas de paramètre configurable        |

### Bloc FeatureInteractions

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| (aucun)     |        |                                  | Ce bloc n’a pas de paramètre configurable        |

### Bloc FeatureSelector

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| method      | select | univariate, rfe, model           | Méthode de sélection de features                 |
| k           | number | 1, 5, 10, ...                    | Nombre de features à sélectionner                |

### Bloc PCATransformer

| Paramètre     | Type   | Valeurs possibles / Exemple      | Description                                      |
|---------------|--------|----------------------------------|--------------------------------------------------|
| n_components  | number | 2, 5, 10, ...                    | Nombre de composantes principales à conserver     |
| columns       | text   | col1,col2,... / vide             | Colonnes à utiliser pour la PCA (optionnel)      |

### Bloc PolynomialFeatures

| Paramètre         | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------------|--------|----------------------------------|--------------------------------------------------|
| degree            | number | 2, 3, 4, ...                     | Degré des features polynomiales                  |
| include_bias      | checkbox | true / false                   | Inclure une colonne de biais                     |
| interaction_only  | checkbox | true / false                   | Générer uniquement les interactions              |
| columns           | text   | col1,col2,... / vide             | Colonnes à transformer (vide = toutes)           |

### Bloc TargetEncoder

| Paramètre           | Type   | Valeurs possibles / Exemple      | Description                                      |
|---------------------|--------|----------------------------------|--------------------------------------------------|
| target_column       | text   | nom_colonne                      | Colonne cible pour l’encodage                    |
| categorical_columns | text   | col1,col2,... / vide             | Colonnes catégorielles à encoder                 |
| smoothing           | number | 1.0, 2.0, ...                    | Lissage de l’encodage (anti-overfitting)         |

### Bloc DecisionTree

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| task          | select   | classification, regression          | Type de tâche (classification ou régression)     |
| target_column | text     | nom_colonne                         | Colonne cible                                    |

### Bloc RandomForest

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| task_type     | select   | classification, regression          | Type de tâche                                    |
| target_column | text     | nom_colonne                         | Colonne cible                                    |
| n_estimators  | number   | 10, 100, 200                        | Nombre d’arbres                                  |
| max_depth     | number   | 5, 10, None                         | Profondeur maximale                              |
| random_state  | number   | 42, ...                             | Graine aléatoire                                 |
| max_features  | select   | sqrt, log2, None                    | Nombre max de features par split                 |

### Bloc LogisticRegression

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| target_column | text     | nom_colonne                         | Colonne cible                                    |
| max_iter      | number   | 100, 1000, ...                      | Nombre maximum d’itérations                      |
| C             | number   | 0.1, 1.0, 10.0                      | Inverse de la régularisation                     |
| solver        | select   | lbfgs, saga, liblinear, newton-cg   | Algorithme d’optimisation                        |
| random_state  | number   | 42, ...                             | Graine aléatoire                                 |

### Bloc KNN

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| task          | select   | classification, regression          | Type de tâche                                    |
| n_neighbors   | number   | 3, 5, 10                            | Nombre de voisins                                |
| target_column | text     | nom_colonne                         | Colonne cible (optionnel selon usage)            |

### Bloc NeuralNetwork

| Paramètre           | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------------|----------|-------------------------------------|--------------------------------------------------|
| task                | select   | classification, regression          | Type de tâche                                    |
| hidden_layer_sizes  | text     | 100, (100,50), (64,32,16)           | Taille des couches cachées                       |
| max_iter            | number   | 100, 500, 1000                      | Nombre maximum d’itérations                      |

### Bloc LinearRegression

| Paramètre        | Type     | Valeurs possibles / Exemple         | Description                                      |
|------------------|----------|-------------------------------------|--------------------------------------------------|
| regression_type  | select   | linear, ridge, lasso                | Type de régression                               |
| alpha            | number   | 0.1, 1.0, 10.0                      | Facteur de régularisation (Ridge/Lasso)          |
| target_column    | text     | nom_colonne                         | Colonne cible                                    |
| fit_intercept    | checkbox | true / false                        | Ajuster l’ordonnée à l’origine                   |
| normalize        | checkbox | true / false                        | Normaliser les variables explicatives            |

### Bloc GradientBoosting

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| task          | select   | classification, regression          | Type de tâche                                    |
| n_estimators  | number   | 100, 200                            | Nombre d’itérations (arbres)                     |
| learning_rate | number   | 0.01, 0.1, 0.2                      | Taux d’apprentissage                             |

### Bloc NaiveBayes

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| algo          | select   | gaussian, multinomial               | Type d’algorithme Naive Bayes                    |
| target_column | text     | nom_colonne                         | Colonne cible                                    |

### Bloc SVM

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| task          | select   | classification, regression, oneclass| Type de tâche                                    |
| target_column | text     | nom_colonne                         | Colonne cible                                    |

### Bloc RegularizedRegression

| Paramètre        | Type     | Valeurs possibles / Exemple         | Description                                      |
|------------------|----------|-------------------------------------|--------------------------------------------------|
| regression_type  | select   | ridge, lasso, elastic_net           | Type de régression réguliarisée                  |
| target_column    | text     | nom_colonne                         | Colonne cible                                    |
| alpha            | number   | 0.1, 1.0, 10.0                      | Facteur de régularisation                        |
| l1_ratio         | number   | 0.0 - 1.0                           | Ratio L1 pour ElasticNet                         |
| max_iter         | number   | 100, 1000, ...                      | Nombre maximum d’itérations                      |

### Bloc KMeansClustering

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| n_clusters    | number   | 2, 3, 8, ...                        | Nombre de clusters                               |
| auto_optimize | checkbox | true / false                        | Optimisation automatique du nombre de clusters   |
| columns       | text     | col1,col2,... / vide                | Colonnes à utiliser (optionnel)                  |

### Bloc DBSCAN

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| eps           | number   | 0.1, 0.5, 1.0                       | Distance maximale entre deux points              |
| min_samples   | number   | 3, 5, 10                            | Nombre min. d’échantillons dans un voisinage     |
| scale_features| checkbox | true / false                        | Normaliser les features                          |
| columns       | text     | col1,col2,... / vide                | Colonnes à utiliser (optionnel)                  |

### Bloc HierarchicalClustering

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| n_clusters    | number   | 2, 3, 5, ...                        | Nombre de clusters                               |

### Bloc GaussianMixture

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| n_components  | number   | 2, 3, 5, ...                        | Nombre de composantes (clusters)                 |

### Bloc AnomalyDetection

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| algo          | select   | isolation_forest, oneclass_svm      | Algorithme de détection d’anomalies              |

### Bloc AssociationRules

| Paramètre      | Type     | Valeurs possibles / Exemple         | Description                                      |
|---------------|----------|-------------------------------------|--------------------------------------------------|
| min_support   | number   | 0.1, 0.5, 0.8                       | Support minimal pour les itemsets fréquents      |
| min_confidence| number   | 0.5, 0.7, 0.9                       | Confiance minimale pour les règles               |

### Bloc ClassificationMetrics

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| y_true      | array  | [0,1,1,0], ...                   | Vraies classes                                   |
| y_pred      | array  | [0,1,0,0], ...                   | Prédictions                                      |
| true_column | text   | nom_colonne                      | Colonne des vraies classes (optionnel)           |
| pred_column | text   | nom_colonne                      | Colonne des prédictions (optionnel)              |

### Bloc RegressionMetrics

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| y_true      | array  | [1.2,2.3,3.1], ...               | Valeurs réelles                                  |
| y_pred      | array  | [1.1,2.5,3.0], ...               | Prédictions                                      |
| true_column | text   | nom_colonne                      | Colonne des vraies valeurs (optionnel)           |
| pred_column | text   | nom_colonne                      | Colonne des prédictions (optionnel)              |

### Bloc ConfusionMatrix

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| y_true      | array  | [0,1,1,0], ...                   | Vraies classes                                   |
| y_pred      | array  | [0,1,0,0], ...                   | Prédictions                                      |

### Bloc CrossValidation

| Paramètre      | Type   | Valeurs possibles / Exemple      | Description                                      |
|----------------|--------|----------------------------------|--------------------------------------------------|
| estimator      | object | Modèle scikit-learn              | Estimateur à valider                             |
| X              | array  | Données d’entrée                 | Features                                         |
| y              | array  | Labels                           | Labels                                           |
| scoring        | text   | accuracy, f1, r2, ...            | Métrique de scoring                              |
| cv             | number | 3, 5, 10                         | Nombre de folds                                  |
| target_column  | text   | nom_colonne                      | Colonne cible (optionnel)                        |
| cv_folds       | number | 3, 5, 10                         | Nombre de folds (optionnel)                      |

### Bloc FeatureImportance

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| model       | object | Modèle entraîné                  | Modèle à expliquer                               |

### Bloc LearningCurves

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| estimator   | object | Modèle scikit-learn              | Estimateur à analyser                            |
| X           | array  | Données d’entrée                 | Features                                         |
| y           | array  | Labels                           | Labels                                           |

### Bloc PredictionVisualizer

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| y_true      | array  | [1.2,2.3,3.1], ...               | Valeurs réelles                                  |
| y_pred      | array  | [1.1,2.5,3.0], ...               | Prédictions                                      |

### Bloc SHAPExplainer

| Paramètre   | Type   | Valeurs possibles / Exemple      | Description                                      |
|-------------|--------|----------------------------------|--------------------------------------------------|
| model       | object | Modèle entraîné                  | Modèle à expliquer                               |

### Bloc CorrelationAnalysis

| Paramètre         | Type     | Valeurs possibles / Exemple         | Description                                      |
|-------------------|----------|-------------------------------------|--------------------------------------------------|
| method            | select   | pearson, spearman, kendall          | Méthode de corrélation                           |
| threshold         | number   | 0.7, 0.8, 0.9                       | Seuil de corrélation élevée                      |
| remove_high_corr  | checkbox | true / false                        | Supprimer les features hautement corrélées        |

### Bloc ARIMAModel

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| col         | text     | nom_colonne                      | Colonne à modéliser                              |
| order       | text     | (1,1,1), (2,1,2)                 | Paramètres ARIMA (p,d,q)                         |

### Bloc TimeSeriesDecomposition

| Paramètre         | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------------|----------|----------------------------------|--------------------------------------------------|
| time_column       | text     | nom_colonne                      | Colonne temporelle                               |
| value_column      | text     | nom_colonne                      | Colonne des valeurs                              |
| model             | select   | additive, multiplicative         | Type de modèle de décomposition                  |
| period            | number   | 12, 24, ...                      | Période saisonnière                              |
| extrapolate_trend | text     | freq, None                       | Extrapolation de la tendance                     |

### Bloc LagFeatures

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| lags        | text     | 1, 3, 5, ...                     | Liste des retards à créer                        |
| col         | text     | nom_colonne                      | Colonne à décaler                                |

### Bloc RollingStatistics

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| window      | number   | 3, 7, 30                         | Taille de la fenêtre mobile                      |
| col         | text     | nom_colonne                      | Colonne à analyser                               |

### Bloc SeasonalityDecomposer

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| col         | text     | nom_colonne                      | Colonne à décomposer                             |
| period      | number   | 12, 24, ...                      | Période saisonnière                              |

### Bloc TimeSeriesCrossValidator

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| n_splits    | number   | 3, 5, 10                         | Nombre de splits temporels                        |

### Bloc TimeSeriesLoader

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| path        | file     | .csv, .tsv                       | Chemin du fichier à charger                      |
| date_col    | text     | nom_colonne                      | Colonne de dates                                 |

### Bloc CustomCodeBlock

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| func        | object   | fonction Python                  | Fonction personnalisée à appliquer               |

### Bloc ImageLoader

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| folder      | folder   | ./images, ./data/imgs            | Dossier contenant les images                     |
| size        | text     | (224,224), (128,128)             | Taille de redimensionnement                      |

### Bloc ModelEnsembler

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| method      | select   | voting, stacking                 | Méthode d’ensemble                               |
| estimators  | object   | liste de tuples (nom, modèle)    | Estimateurs à combiner                           |

### Bloc PipelineCombiner

| Paramètre   | Type     | Valeurs possibles / Exemple      | Description                                      |
|-------------|----------|----------------------------------|--------------------------------------------------|
| pipelines   | object   | liste de pipelines               | Pipelines à combiner                             |

### Bloc TFIDFVectorizer

| Paramètre      | Type     | Valeurs possibles / Exemple      | Description                                      |
|----------------|----------|----------------------------------|--------------------------------------------------|
| col            | text     | nom_colonne                      | Colonne texte à vectoriser                       |
| text_column    | text     | nom_colonne                      | Colonne texte (optionnel)                        |
| max_features   | number   | 1000, 5000                       | Nombre max de features (optionnel)               |

### Bloc TextPreprocessor

| Paramètre      | Type     | Valeurs possibles / Exemple      | Description                                      |
|----------------|----------|----------------------------------|--------------------------------------------------|
| col            | text     | nom_colonne                      | Colonne texte à prétraiter                       |
| text_column    | text     | nom_colonne                      | Colonne texte (optionnel)                        |


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