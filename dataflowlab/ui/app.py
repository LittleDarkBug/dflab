import gradio as gr
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

from dataflowlab.core.block_registry import BlockRegistry
from dataflowlab.core.pipeline import Pipeline
# Import conditionnel du LLM Assistant
try:
    from dataflowlab.llm.assistant import LLMAssistant
    LLM_AVAILABLE = True
except ImportError:
    LLMAssistant = None
    LLM_AVAILABLE = False
    
from dataflowlab.eda.auto_eda import generate_eda_report
from dataflowlab.export.notebook_exporter import export_pipeline_to_notebook
from dataflowlab.utils.logger import get_logger

logger = get_logger("UI")

# Auto-découverte des blocs au démarrage
registry = BlockRegistry()

def get_block_types() -> List[str]:
    """Retourne la liste des types de blocs disponibles."""
    registry = BlockRegistry()
    return list(registry.blocks.keys())

def get_block_categories() -> Dict[str, List[str]]:
    """Retourne les blocs organisés par catégorie."""
    registry = BlockRegistry()
    return registry.categories

def create_pipeline_visualization(pipeline_blocks: List[Dict]) -> go.Figure:
    """Crée une visualisation du pipeline sous forme de graphique."""
    if not pipeline_blocks:
        fig = go.Figure()
        fig.add_annotation(
            text="Pipeline vide - Ajoutez des blocs",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor="center", yanchor="middle",
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title="Visualisation du Pipeline",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    
    # Création du graphique de flux
    labels = []
    colors = []
    
    # Couleurs par catégorie
    color_map = {
        "data_input": "#3498db",
        "data_cleaning": "#e74c3c", 
        "feature_engineering": "#f39c12",
        "supervised": "#2ecc71",
        "unsupervised": "#9b59b6",
        "evaluation": "#1abc9c",
        "advanced": "#34495e"
    }
    
    for i, block in enumerate(pipeline_blocks):
        block_type = block["type"]
        # Trouver la catégorie du bloc
        category = "other"
        for cat, blocks in get_block_categories().items():
            if block_type in blocks:
                category = cat
                break
        
        labels.append(f"{i+1}. {block_type}")
        colors.append(color_map.get(category, "#95a5a6"))
    
    # Création des liens entre les blocs
    source = list(range(len(labels)-1))
    target = list(range(1, len(labels)))
    values = [1] * len(source)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color="rgba(128,128,128,0.3)"
        )
    )])
    
    fig.update_layout(
        title="Pipeline DataFlowLab",
        font_size=12,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def get_block_config_interface(block_type: str) -> List[gr.components.Component]:
    """Génère l'interface de configuration dynamique pour un bloc."""
    try:
        registry = BlockRegistry()
        block_cls = registry.blocks.get(block_type)
        if not block_cls:
            return [gr.Textbox(label="Erreur", value=f"Bloc {block_type} non trouvé", interactive=False)]
        
        components = []
        
        # Configuration spécifique par type de bloc
        if block_type in ['CSVLoader', 'ExcelLoader']:
            components.extend([
                gr.Markdown(f"### 📁 Configuration {block_type}"),
                gr.File(
                    label="� Sélectionner le fichier",
                    file_types=[".csv", ".xlsx", ".xls"] if block_type == "ExcelLoader" else [".csv"],
                    elem_id=f"file_input_{block_type}"
                ),
                gr.Textbox(
                    label="Séparateur (CSV)", 
                    value=",", 
                    visible=(block_type == "CSVLoader"),
                    placeholder="Exemple: ; | \t",
                    elem_id=f"separator_{block_type}"
                ),
                gr.Dropdown(
                    label="Encodage",
                    choices=["utf-8", "latin-1", "cp1252", "iso-8859-1"],
                    value="utf-8",
                    elem_id=f"encoding_{block_type}"
                ),
                gr.Checkbox(
                    label="Première ligne = en-têtes", 
                    value=True,
                    elem_id=f"header_{block_type}"
                ),
                gr.Number(
                    label="Lignes à ignorer (début)", 
                    value=0, 
                    minimum=0,
                    elem_id=f"skip_rows_{block_type}"
                ),
                gr.Number(
                    label="Nombre max de lignes à lire", 
                    value=None, 
                    minimum=1, 
                    placeholder="Toutes",
                    elem_id=f"max_rows_{block_type}"
                ),
                gr.Markdown("✅ **Utilisation :** Sélectionnez votre fichier ci-dessus, puis cliquez sur 'Ajouter au Pipeline'")
            ])
        
        elif block_type == 'JSONLoader':
            components.extend([
                gr.Markdown("### 📋 Configuration JSONLoader"),
                gr.File(
                    label="📂 Sélectionner le fichier JSON",
                    file_types=[".json"],
                    elem_id=f"file_input_{block_type}"
                ),
                gr.Textbox(
                    label="Chemin vers les données (optionnel)", 
                    placeholder="data.records ou data.items",
                    info="Chemin JsonPath pour extraire les données du JSON",
                    elem_id=f"json_path_{block_type}"
                ),
                gr.Dropdown(
                    label="Orientation des données",
                    choices=["records", "index", "values", "split"],
                    value="records",
                    elem_id=f"orientation_{block_type}"
                ),
                gr.Markdown("✅ **Utilisation :** Sélectionnez votre fichier JSON ci-dessus")
            ])
        
        elif block_type == 'SQLConnector':
            components.extend([
                gr.Markdown("### 🗃️ Configuration SQLConnector"),
                gr.Textbox(
                    label="Chaîne de connexion", 
                    placeholder="sqlite:///database.db",
                    elem_id=f"connection_{block_type}"
                ),
                gr.Textbox(
                    label="Requête SQL", 
                    placeholder="SELECT * FROM table", 
                    lines=3,
                    elem_id=f"query_{block_type}"
                ),
                gr.Textbox(
                    label="Utilisateur (optionnel)", 
                    placeholder="username",
                    elem_id=f"username_{block_type}"
                ),
                gr.Textbox(
                    label="Mot de passe (optionnel)", 
                    type="password",
                    elem_id=f"password_{block_type}"
                )
            ])
        
        else:
            # Configuration JSON générique pour les autres blocs
            components.extend([
                gr.Markdown(f"### ⚙️ Configuration {block_type}"),
                gr.Textbox(
                    label="⚙️ Paramètres JSON",
                    value="{}",
                    placeholder='{"param1": "value1", "param2": 123}',
                    lines=5,
                    info="Configuration au format JSON",
                    elem_id=f"json_config_{block_type}"
                ),
                gr.Markdown("💡 **Exemple de configuration :**"),
                gr.Code(
                    value=get_example_config(block_type),
                    language="json",
                    label="Exemple",
                    interactive=False
                )
            ])
        
        # Ajouter toujours une description du bloc
        block_description = get_block_description(block_type)
        if block_description:
            components.insert(0, gr.Markdown(f"**Description :** {block_description}"))
        
        return components
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'interface pour {block_type}: {e}")
        return [gr.Textbox(label="Erreur", value=str(e), interactive=False)]

def get_example_config(block_type: str) -> str:
    """Retourne un exemple de configuration JSON pour un bloc."""
    examples = {
        "LinearRegressionBlock": '{\n  "target_column": "price",\n  "test_size": 0.2,\n  "random_state": 42\n}',
        "RandomForestBlock": '{\n  "target_column": "species",\n  "n_estimators": 100,\n  "max_depth": 10,\n  "random_state": 42\n}',
        "KMeansClustering": '{\n  "n_clusters": 3,\n  "random_state": 42\n}',
        "FeatureScaler": '{\n  "scaler_type": "StandardScaler",\n  "columns": ["col1", "col2"]\n}',
        "MissingValuesHandler": '{\n  "strategy": "mean",\n  "columns": ["numeric_col1", "numeric_col2"]\n}'
    }
    return examples.get(block_type, '{\n  "param1": "value1",\n  "param2": 123\n}')

def get_block_description(block_type: str) -> str:
    """Retourne une description du bloc."""
    descriptions = {
        "CSVLoader": "📊 Charge des données depuis un fichier CSV avec options avancées d'encodage",
        "ExcelLoader": "📈 Charge des données depuis un fichier Excel (.xlsx, .xls)",
        "JSONLoader": "📋 Charge des données depuis un fichier JSON avec parsing configurable",
        "SQLConnector": "🗃️ Se connecte à une base de données et exécute des requêtes SQL",
        "LinearRegressionBlock": "📈 Régression linéaire pour prédictions numériques continues",
        "LogisticRegressionBlock": "🎯 Régression logistique pour classification binaire/multiclasse",
        "RandomForestBlock": "🌳 Forêt aléatoire - algorithme robuste pour classification/régression",
        "GradientBoostingBlock": "⚡ Gradient Boosting - algorithme performant par assemblage de modèles",
        "KMeansClustering": "🎪 Clustering K-means pour grouper des données non-supervisées",
        "DBSCANBlock": "🔍 DBSCAN - clustering basé sur la densité avec détection d'outliers",
        "FeatureScaler": "⚖️ Normalisation/standardisation des variables numériques",
        "PCATransformer": "📊 Analyse en Composantes Principales pour réduction de dimensionnalité",
        "OneHotEncoderBlock": "🏷️ Encodage des variables catégorielles en variables binaires",
        "MissingValuesHandler": "🔧 Traitement intelligent des valeurs manquantes",
        "OutlierDetector": "🚨 Détection et traitement des valeurs aberrantes",
        "DuplicateRemover": "🗂️ Suppression des lignes dupliquées"
    }
    return descriptions.get(block_type, "🔧 Bloc de traitement des données")

def create_main_interface():
    """Crée l'interface principale de DataFlowLab."""
    
    with gr.Blocks(
        title="DataFlowLab - Plateforme ML Visuelle",
        theme=gr.themes.Soft(),
        css="""
        .block-container { 
            border: 2px dashed #ccc; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 10px;
            min-height: 200px;
        }
        .pipeline-block {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin: 5px;
        }
        """
    ) as demo:
        
        # En-tête
        gr.Markdown("""
        # DataFlowLab
        ## Plateforme Visuelle de Pipelines Machine Learning
        Créez, testez et déployez vos pipelines ML par drag-and-drop
        """)
        
        # État global du pipeline
        pipeline_state = gr.State([])
        current_data = gr.State(None)
        
        with gr.Tabs():
            
            # Tab 1: Construction du Pipeline
            with gr.Tab("Pipeline Builder"):
                with gr.Row():
                    # Panneau gauche: Bibliothèque de blocs
                    with gr.Column(scale=1):
                        gr.Markdown("### Bibliothèque de Blocs")
                        
                        # Blocs organisés par catégorie
                        categories = get_block_categories()
                        
                        # Variables pour stocker les boutons de blocs
                        block_buttons = {}
                        
                        for category, blocks in categories.items():
                            with gr.Accordion(f"📦 {category.title()}", open=category=="data_input"):
                                gr.Markdown("*Clic = configuration du bloc*")
                                for block_type in blocks:
                                    btn = gr.Button(
                                        value=f"⚙️ {block_type}",
                                        variant="secondary",
                                        size="sm",
                                        elem_id=f"add_{block_type}"
                                    )
                                    block_buttons[block_type] = btn
                    
                    # Panneau central: Zone de construction du pipeline
                    with gr.Column(scale=2):
                        gr.Markdown("### Construction du Pipeline")
                        
                        # Visualisation graphique du pipeline
                        pipeline_viz = gr.Plot(
                            label="Visualisation du Pipeline",
                            value=create_pipeline_visualization([])
                        )
                        
                        # Liste des blocs du pipeline
                        with gr.Row():
                            pipeline_list = gr.Dataframe(
                                headers=["#", "Type", "Paramètres", "Status"],
                                datatype=["number", "str", "str", "str"],
                                label="Blocs du Pipeline",
                                interactive=False
                            )
                        
                        # Contrôles du pipeline
                        with gr.Row():
                            run_pipeline_btn = gr.Button("▶️ Exécuter Pipeline", variant="primary")
                            validate_pipeline_btn = gr.Button("✅ Valider Pipeline")
                            clear_pipeline_btn = gr.Button("🗑️ Vider Pipeline", variant="stop")
                    
                    # Panneau droit: Configuration automatique
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration du Bloc")
                        
                        gr.Markdown("""
                        **Flow d'utilisation :**
                        1. Cliquez sur un bloc dans la bibliothèque
                        2. Configurez ses paramètres ici
                        3. Ajoutez au pipeline
                        """)
                        
                        # Bloc actuellement sélectionné
                        current_block_type = gr.State(None)
                        selected_block_name = gr.Textbox(
                            label="Bloc sélectionné",
                            value="Aucun bloc sélectionné",
                            interactive=False
                        )
                        
                        # Zone de configuration avec composants statiques
                        config_area = gr.Column(visible=False)
                        with config_area:
                            gr.Markdown("#### Configuration")
                            
                            # Composants pour les blocs data_input
                            with gr.Group(visible=False) as data_input_config:
                                gr.Markdown("### 📁 Configuration Fichier")
                                file_input = gr.File(label="📂 Sélectionner le fichier")
                                separator_input = gr.Textbox(label="Séparateur", value=",", visible=False)
                                encoding_input = gr.Dropdown(
                                    label="Encodage",
                                    choices=["utf-8", "latin-1", "cp1252"],
                                    value="utf-8"
                                )
                                header_input = gr.Checkbox(label="Première ligne = en-têtes", value=True)
                            
                            # Configuration JSON générique pour autres blocs
                            with gr.Group(visible=False) as generic_config:
                                gr.Markdown("### ⚙️ Configuration JSON")
                                json_config = gr.Textbox(
                                    label="Paramètres JSON",
                                    value="{}",
                                    lines=5,
                                    placeholder='{"param1": "value1", "param2": 123}'
                                )
                                config_example = gr.Code(
                                    value='{"example": "configuration"}',
                                    language="json",
                                    label="Exemple",
                                    interactive=False
                                )
                        
                        # Boutons d'action
                        with gr.Row():
                            add_default_btn = gr.Button(
                                "⚡ Ajouter par Défaut",
                                variant="secondary",
                                visible=False
                            )
                            add_configured_btn = gr.Button(
                                "✅ Ajouter avec Config",
                                variant="primary",
                                visible=False
                            )
                
                # Zone de résultats et logs
                with gr.Row():
                    with gr.Column():
                        pipeline_output = gr.Textbox(
                            label="Sortie du Pipeline",
                            lines=10,
                            interactive=False
                        )
                    
                    with gr.Column():
                        pipeline_logs = gr.Textbox(
                            label="Logs d'Exécution",
                            lines=10,
                            interactive=False
                        )
            
            # Tab 2: EDA Automatique
            with gr.Tab("Analyse Exploratoire"):
                gr.Markdown("### Analyse Exploratoire Automatique")
                
                with gr.Row():
                    with gr.Column():
                        data_upload = gr.File(
                            label="Charger des données (CSV/Excel)",
                            file_types=[".csv", ".xlsx", ".xls"]
                        )
                        run_eda_btn = gr.Button("🔍 Lancer EDA", variant="primary")
                    
                    with gr.Column():
                        eda_summary = gr.Textbox(
                            label="Résumé EDA",
                            lines=5,
                            interactive=False
                        )
                
                # Visualisations EDA
                with gr.Row():
                    eda_plots = gr.Plot(label="Visualisations EDA")
                
                eda_report = gr.HTML(label="Rapport EDA Complet")
            
            # Tab 3: Export et Déploiement
            with gr.Tab("Export & Code"):
                gr.Markdown("### Export et Génération de Code")
                
                with gr.Row():
                    with gr.Column():
                        export_format = gr.Radio(
                            choices=["Python Script", "Jupyter Notebook", "Pipeline Config"],
                            label="Format d'export",
                            value="Python Script"
                        )
                        
                        export_btn = gr.Button("📄 Générer Code", variant="primary")
                        download_btn = gr.Button("💾 Télécharger", variant="secondary")
                    
                    with gr.Column():
                        save_pipeline_btn = gr.Button("💾 Sauvegarder Pipeline")
                        load_pipeline_btn = gr.Button("📂 Charger Pipeline") 
                        
                        pipeline_file = gr.File(
                            label="Fichier de pipeline (.json)",
                            file_types=[".json"]
                        )
                
                generated_code = gr.Code(
                    label="Code Généré",
                    language="python",
                    lines=20
                )
            
            # Tab 4: Assistant LLM
            with gr.Tab("Assistant IA"):
                gr.Markdown("### Assistant LLM Local")
                
                chatbot = gr.Chatbot(
                    label="Conversation avec l'Assistant",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Votre question",
                        placeholder="Demandez de l'aide sur votre pipeline...",
                        scale=4
                    )
                    send_btn = gr.Button("Envoyer", scale=1, variant="primary")
                
                gr.Examples(
                    examples=[
                        "Comment améliorer les performances de mon modèle ?",
                        "Quels blocs recommandez-vous pour des données textuelles ?",
                        "Expliquez-moi les résultats de mon pipeline",
                        "Comment gérer les valeurs manquantes ?"
                    ],
                    inputs=msg_input
                )
        
        # Fonctions de callback
        def update_pipeline_viz(blocks):
            return create_pipeline_visualization(blocks)
        
        def update_block_config(block_type):
            if block_type:
                components = get_block_config_interface(block_type)
                return [
                    gr.update(visible=True),  # config_area
                    gr.update(visible=True),  # add_configured_block_btn
                    gr.update(children=components)  # config_components
                ]
            return [
                gr.update(visible=False),
                gr.update(visible=False), 
                gr.update(children=[])
            ]
        
        def add_block_to_pipeline(current_blocks, block_type, config_params=None):
            """Ajoute un bloc au pipeline avec sa configuration."""
            try:
                # Si config_params est un dictionnaire, l'utiliser directement
                if isinstance(config_params, dict):
                    params = config_params
                # Sinon, configuration par défaut vide
                else:
                    params = {}
                
                # Validation des dépendances du pipeline
                validation_result = validate_pipeline_dependencies(current_blocks, block_type)
                if not validation_result["valid"]:
                    return current_blocks, [], f"⚠️ {validation_result['message']}"
                
                # Création du bloc avec données transmises du bloc précédent
                new_block = {
                    "id": len(current_blocks) + 1,
                    "type": block_type,
                    "params": params,
                    "status": "Configuré" if params else "Défaut",
                    "input_data": None,  # Sera rempli lors de l'exécution
                    "output_data": None
                }
                
                # Ajout au pipeline
                updated_blocks = current_blocks + [new_block]
                
                # Mise à jour de la liste pour affichage
                pipeline_data = []
                for block in updated_blocks:
                    params_str = json.dumps(block["params"], indent=2) if block["params"] else "{}"
                    pipeline_data.append([
                        block["id"],
                        block["type"],
                        params_str[:100] + "..." if len(params_str) > 100 else params_str,
                        block["status"]
                    ])
                
                config_msg = "avec configuration" if params else "avec paramètres par défaut"
                return updated_blocks, pipeline_data, f"✅ Bloc {block_type} ajouté au pipeline {config_msg}"
                
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout du bloc {block_type}: {e}")
                return current_blocks, [], f"❌ Erreur: {str(e)}"

        def validate_pipeline_dependencies(current_blocks, new_block_type):
            """Valide les dépendances entre blocs dans le pipeline."""
            block_categories = get_block_categories()
            
            # Trouver la catégorie du nouveau bloc
            new_block_category = None
            for category, blocks in block_categories.items():
                if new_block_type in blocks:
                    new_block_category = category
                    break
            
            if not new_block_category:
                return {"valid": False, "message": f"Catégorie inconnue pour {new_block_type}"}
            
            # Règles de dépendances
            dependencies = {
                "data_cleaning": ["data_input"],
                "feature_engineering": ["data_input"],
                "supervised": ["data_input", "feature_engineering"],
                "unsupervised": ["data_input"],
                "evaluation": ["supervised", "unsupervised"]
            }
            
            if new_block_category in dependencies:
                required_categories = dependencies[new_block_category]
                existing_categories = []
                
                for block in current_blocks:
                    for cat, blocks in block_categories.items():
                        if block["type"] in blocks:
                            existing_categories.append(cat)
                            break
                
                missing_deps = []
                for req_cat in required_categories:
                    if req_cat not in existing_categories:
                        missing_deps.append(req_cat)
                
                if missing_deps:
                    return {
                        "valid": False, 
                        "message": f"Dépendances manquantes: {', '.join(missing_deps)}"
                    }
            
            return {"valid": True, "message": "Dépendances validées"}
        
        def execute_pipeline(blocks, current_data_state):
            """Exécute le pipeline de blocs avec transmission des données."""
            if not blocks:
                return "❌ Pipeline vide", "Aucun bloc à exécuter", None
            
            try:
                registry = BlockRegistry()
                logs = []
                current_data = current_data_state
                execution_results = []
                
                for i, block_config in enumerate(blocks):
                    block_type = block_config["type"]
                    block_params = block_config.get("params", {})
                    
                    logs.append(f"🔄 Étape {i+1}: Exécution du bloc {block_type}")
                    logs.append(f"   📋 Paramètres: {json.dumps(block_params, indent=2)}")
                    
                    try:
                        # Obtenir la classe du bloc
                        block_cls = registry.blocks.get(block_type)
                        if not block_cls:
                            raise ValueError(f"Bloc {block_type} non trouvé dans le registre")
                        
                        # Créer une instance du bloc
                        block_instance = block_cls(params=block_params)
                        
                        # Exécuter le bloc avec les données courantes
                        if hasattr(block_instance, 'process'):
                            if current_data is not None:
                                output_data = block_instance.process(current_data)
                                logs.append(f"   ✅ Bloc traité - Input shape: {getattr(current_data, 'shape', 'N/A')} -> Output shape: {getattr(output_data, 'shape', 'N/A')}")
                            else:
                                # Pour les blocs d'entrée, pas de données d'entrée
                                output_data = block_instance.process()
                                logs.append(f"   ✅ Données chargées - Shape: {getattr(output_data, 'shape', 'N/A')}")
                            
                            # Mettre à jour les données courantes pour le bloc suivant
                            current_data = output_data
                            
                            # Stocker les résultats
                            execution_results.append({
                                "block_id": i + 1,
                                "block_type": block_type,
                                "data_shape": getattr(output_data, 'shape', None),
                                "data_type": type(output_data).__name__,
                                "status": "success"
                            })
                            
                            # Mettre à jour le statut du bloc
                            blocks[i]['status'] = 'Exécuté ✅'
                            blocks[i]['output_data'] = output_data
                            
                        else:
                            logs.append(f"   ⚠️ Bloc {block_type} n'a pas de méthode process")
                            execution_results.append({
                                "block_id": i + 1,
                                "block_type": block_type,
                                "status": "warning",
                                "message": "Pas de méthode process"
                            })
                    
                    except Exception as block_error:
                        error_msg = f"Erreur dans le bloc {block_type}: {str(block_error)}"
                        logs.append(f"   ❌ {error_msg}")
                        execution_results.append({
                            "block_id": i + 1,
                            "block_type": block_type,
                            "status": "error",
                            "message": str(block_error)
                        })
                        blocks[i]['status'] = 'Erreur ❌'
                        # Arrêter l'exécution en cas d'erreur
                        break
                
                # Résumé d'exécution
                successful_blocks = sum(1 for r in execution_results if r["status"] == "success")
                
                if current_data is not None:
                    if hasattr(current_data, 'shape'):
                        summary = f"""🎯 Pipeline exécuté avec succès!
                        
📊 Résultats finaux:
• {successful_blocks}/{len(blocks)} blocs exécutés avec succès
• Données finales: {current_data.shape}
• Type: {type(current_data).__name__}

📈 Résumé par étape:"""
                        
                        for result in execution_results:
                            if result["status"] == "success":
                                summary += f"\n✅ {result['block_type']}: {result.get('data_shape', 'N/A')}"
                            elif result["status"] == "error":
                                summary += f"\n❌ {result['block_type']}: {result['message']}"
                            else:
                                summary += f"\n⚠️ {result['block_type']}: {result.get('message', 'Warning')}"
                    else:
                        summary = f"Pipeline exécuté: {successful_blocks}/{len(blocks)} blocs réussis"
                else:
                    summary = f"Pipeline partiellement exécuté: {successful_blocks}/{len(blocks)} blocs réussis"
                
                log_text = "\n".join(logs)
                
                return summary, log_text, current_data
                
            except Exception as e:
                error_msg = f"❌ Erreur d'exécution globale: {str(e)}"
                logger.error(error_msg)
                return error_msg, f"Erreur globale: {str(e)}", None
        
        def clear_pipeline():
            """Vide le pipeline et remet à zéro l'état."""
            empty_viz = create_pipeline_visualization([])
            return [], [], "🗑️ Pipeline vidé", "", None, empty_viz
        
        def validate_pipeline(blocks):
            """Valide la configuration du pipeline."""
            if not blocks:
                return "❌ Pipeline vide"
            
            issues = []
            for i, block in enumerate(blocks):
                if not block.get('type'):
                    issues.append(f"Bloc {i+1}: Type manquant")
                if not isinstance(block.get('params'), dict):
                    issues.append(f"Bloc {i+1}: Paramètres invalides")
            
            if issues:
                return "⚠️ Problèmes détectés:\n" + "\n".join(issues)
            else:
                return f"✅ Pipeline valide ({len(blocks)} blocs)"
        
        # Callback pour sélectionner un bloc depuis la bibliothèque
        def select_block_for_config(block_type):
            """Sélectionne un bloc et affiche son interface de configuration."""
            logger.info(f"Bloc sélectionné pour configuration: {block_type}")
            
            try:
                # Déterminer le type de configuration nécessaire
                is_data_input = block_type in ['CSVLoader', 'ExcelLoader', 'JSONLoader']
                
                # Afficher le séparateur pour CSV seulement
                show_separator = block_type == 'CSVLoader'
                
                # Mettre à jour l'exemple de configuration JSON
                example_config = get_example_config(block_type)
                
                return [
                    block_type,  # current_block_type
                    f"🔧 {block_type} - Prêt pour configuration",  # selected_block_name
                    gr.update(visible=True),  # config_area
                    gr.update(visible=is_data_input),  # data_input_config
                    gr.update(visible=not is_data_input),  # generic_config
                    gr.update(visible=True),  # add_default_btn
                    gr.update(visible=True)   # add_configured_btn
                ]
                
            except Exception as e:
                logger.error(f"Erreur lors de la sélection du bloc {block_type}: {e}")
                return [
                    None,
                    "Erreur lors de la sélection",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                ]
        
        # Connecter les boutons de la bibliothèque à la sélection
        for block_type, btn in block_buttons.items():
            btn.click(
                lambda bt=block_type: select_block_for_config(bt),
                outputs=[current_block_type, selected_block_name, config_area, data_input_config, generic_config, add_default_btn, add_configured_btn]
            )
        
        # Callback pour ajouter avec valeurs par défaut
        def add_block_default(blocks, block_type):
            """Ajoute un bloc avec paramètres par défaut."""
            if not block_type:
                return blocks, [], "⚠️ Aucun bloc sélectionné", gr.update(), None, "Aucun bloc sélectionné", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            updated_blocks, pipeline_data, message = add_block_to_pipeline(blocks, block_type, {})
            viz_update = update_pipeline_viz(updated_blocks)
            
            # Réinitialiser la sélection après ajout
            return updated_blocks, pipeline_data, f"{message} (par défaut)", viz_update, None, "Aucun bloc sélectionné", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Callback pour ajouter avec configuration
        def add_block_configured(blocks, block_type, file_obj, separator, encoding, header, json_config_str):
            """Ajoute un bloc avec configuration."""
            if not block_type:
                return blocks, [], "⚠️ Aucun bloc sélectionné", gr.update(), None, "Aucun bloc sélectionné", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            # Construire la configuration selon le type de bloc
            config_params = {}
            
            # Pour les blocs data_input, utiliser les valeurs du formulaire
            if block_type in ['CSVLoader', 'ExcelLoader', 'JSONLoader']:
                if file_obj and hasattr(file_obj, 'name'):
                    config_params["file_path"] = file_obj.name
                    
                if block_type == 'CSVLoader':
                    config_params["separator"] = separator if separator else ","
                    
                config_params["encoding"] = encoding if encoding else "utf-8"
                config_params["header"] = 0 if header else None
            
            # Pour les autres blocs, utiliser la configuration JSON
            else:
                if json_config_str and json_config_str.strip() != "{}":
                    try:
                        config_params = json.loads(json_config_str)
                    except json.JSONDecodeError as e:
                        return blocks, [], f"❌ Erreur JSON: {str(e)}", gr.update(), block_type, f"🔧 {block_type} - Prêt pour configuration", gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            
            updated_blocks, pipeline_data, message = add_block_to_pipeline(blocks, block_type, config_params)
            viz_update = update_pipeline_viz(updated_blocks)
            
            # Réinitialiser la sélection après ajout réussi
            return updated_blocks, pipeline_data, f"{message} (configuré)", viz_update, None, "Aucun bloc sélectionné", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Connecter les boutons d'ajout
        add_default_btn.click(
            add_block_default,
            inputs=[pipeline_state, current_block_type],
            outputs=[pipeline_state, pipeline_list, pipeline_output, pipeline_viz, current_block_type, selected_block_name, config_area, add_default_btn, add_configured_btn]
        )
        
        add_configured_btn.click(
            add_block_configured,
            inputs=[pipeline_state, current_block_type, file_input, separator_input, encoding_input, header_input, json_config],
            outputs=[pipeline_state, pipeline_list, pipeline_output, pipeline_viz, current_block_type, selected_block_name, config_area, add_default_btn, add_configured_btn]
        )
        
        pipeline_state.change(
            update_pipeline_viz,
            inputs=[pipeline_state],
            outputs=[pipeline_viz]
        )
        
        # Callback avancé pour ajouter un bloc configuré
        def add_configured_block_with_params(blocks, block_type):
            """Ajoute un bloc configuré au pipeline."""
            if not block_type:
                return blocks, [], "⚠️ Sélectionnez un type de bloc", gr.update(), None
            
            logger.info(f"Ajout du bloc configuré {block_type}")
            
            # Pour l'instant, ajouter avec paramètres par défaut
            # TODO: Récupérer les valeurs des composants de configuration
            config_params = {}
            
            # Utiliser la fonction add_block_to_pipeline 
            updated_blocks, pipeline_data, message = add_block_to_pipeline(blocks, block_type, config_params)
            
            # Mettre à jour la visualisation
            viz_update = update_pipeline_viz(updated_blocks)
            
            # Réinitialiser la sélection pour permettre une nouvelle configuration
            return updated_blocks, pipeline_data, f"{message} ✨ (Configuré)", viz_update, None
        
        # Fonction pour connecter dynamiquement le callback du bouton "Ajouter au Pipeline"
        def connect_add_button_callback(block_type):
            """Connecte le bouton d'ajout avec les bons inputs selon le type de bloc."""
            if not block_type:
                return
            
            # Cette fonction sera appelée pour reconfigurer le callback quand le type change
            # Pour l'instant, on garde le callback simple
            pass
        
        
        # Callback pour exécuter le pipeline avec transmission de données
        run_pipeline_btn.click(
            execute_pipeline,
            inputs=[pipeline_state, current_data],
            outputs=[pipeline_output, pipeline_logs, current_data]
        )
        
        # Callback pour valider le pipeline
        validate_pipeline_btn.click(
            validate_pipeline,
            inputs=[pipeline_state],
            outputs=[pipeline_output]
        )
        
        # Callback pour vider le pipeline
        clear_pipeline_btn.click(
            clear_pipeline,
            outputs=[pipeline_state, pipeline_list, pipeline_output, pipeline_logs, current_data, pipeline_viz]
        )
        
        # (Les callbacks des boutons de la bibliothèque sont définis plus haut)
        
        # Callback pour ajouter un bloc configuré au pipeline (version améliorée)
        def add_configured_block_advanced(blocks, block_type, *config_values):
            """Ajoute un bloc configuré au pipeline avec tous ses paramètres."""
            if not block_type:
                return blocks, [], "⚠️ Sélectionnez un type de bloc", gr.update()
            
            updated_blocks, pipeline_data, message = add_block_to_pipeline(blocks, block_type, *config_values)
            
            # Mettre à jour la visualisation
            viz_update = update_pipeline_viz(updated_blocks)
            
            return updated_blocks, pipeline_data, message, viz_update
        
        def clear_pipeline():
            """Vide le pipeline et remet à zéro l'état."""
            empty_viz = create_pipeline_visualization([])
            return [], [], "🗑️ Pipeline vidé", "", None, empty_viz
        
        def validate_pipeline(blocks):
            """Valide la configuration du pipeline avec vérifications avancées."""
            if not blocks:
                return "❌ Pipeline vide"
            
            issues = []
            warnings = []
            
            # Vérifications basiques
            for i, block in enumerate(blocks):
                if not block.get('type'):
                    issues.append(f"Bloc {i+1}: Type manquant")
                if not isinstance(block.get('params'), dict):
                    issues.append(f"Bloc {i+1}: Paramètres invalides")
            
            # Vérifications de dépendances
            block_categories = get_block_categories()
            categories_present = []
            
            for block in blocks:
                for cat, cat_blocks in block_categories.items():
                    if block["type"] in cat_blocks:
                        if cat not in categories_present:
                            categories_present.append(cat)
                        break
            
            # Vérifications spécifiques
            if "supervised" in categories_present or "unsupervised" in categories_present:
                if "data_input" not in categories_present:
                    issues.append("Aucun bloc de chargement de données trouvé")
            
            if "evaluation" in categories_present:
                if "supervised" not in categories_present and "unsupervised" not in categories_present:
                    issues.append("Aucun modèle à évaluer trouvé")
            
            # Recommandations
            if "data_input" in categories_present and "data_cleaning" not in categories_present:
                warnings.append("Recommandation: Ajouter des blocs de nettoyage de données")
            
            if "supervised" in categories_present and "evaluation" not in categories_present:
                warnings.append("Recommandation: Ajouter des blocs d'évaluation")
            
            # Compilation du résultat
            result_parts = []
            
            if issues:
                result_parts.append("❌ Problèmes critiques:")
                result_parts.extend([f"  • {issue}" for issue in issues])
            
            if warnings:
                result_parts.append("⚠️ Recommandations:")
                result_parts.extend([f"  • {warning}" for warning in warnings])
            
            if not issues and not warnings:
                result_parts.append(f"✅ Pipeline parfaitement configuré!")
                result_parts.append(f"📊 {len(blocks)} blocs, {len(categories_present)} catégories")
                result_parts.append(f"🎯 Catégories: {', '.join(categories_present)}")
            elif not issues:
                result_parts.append(f"✅ Pipeline valide avec quelques recommandations")
                result_parts.append(f"📊 {len(blocks)} blocs configurés correctement")
            
            return "\n".join(result_parts)
        
        # Les callbacks des boutons de la bibliothèque sont définis plus haut (sélection pour configuration)
    
    return demo

def main():
    """Point d'entrée principal de l'application."""
    try:
        logger.info("Démarrage de DataFlowLab...")
        
        # Découverte automatique des blocs
        logger.info("Découverte des blocs...")
        BlockRegistry.auto_discover_blocks()
        
        registered_blocks = BlockRegistry.list_blocks()
        logger.info(f"Blocs enregistrés: {len(registered_blocks)}")
        
        for category, blocks in BlockRegistry.list_by_category().items():
            logger.info(f"  {category}: {len(blocks)} blocs")
        
        # Création et lancement de l'interface
        demo = create_main_interface()
        
        logger.info("Lancement de l'interface Gradio...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage: {e}")
        raise

def launch_app():
    """Lance l'application DataFlowLab."""
def main():
    """Point d'entrée principal de l'application."""
    try:
        logger.info("Démarrage de DataFlowLab...")
        
        # Découverte automatique des blocs
        logger.info("Découverte des blocs...")
        registry = BlockRegistry()
        
        registered_blocks = registry.blocks
        categories = registry.categories
        
        logger.info(f"Blocs enregistrés: {len(registered_blocks)}")
        logger.info(f"Catégories: {list(categories.keys())}")
        
        for cat, blocks in categories.items():
            logger.info(f"  {cat}: {len(blocks)} blocs")
        
        # Création et lancement de l'interface complète
        demo = create_main_interface()
        
        print("🚀 Lancement de DataFlowLab avec interface complète")
        print(f"📊 {len(registered_blocks)} blocs ML disponibles")
        print(f"🎯 {len(categories)} catégories de blocs")
        
        demo.launch(
            server_port=7861, 
            share=False,
            show_api=False,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage: {e}")
        print(f"❌ Erreur: {e}")
        raise

def launch_simple():
    """Lance une version simplifiée pour tests."""
    try:
        def hello_dataflowlab():
            registry = BlockRegistry()
            return f"DataFlowLab ready! {len(registry.blocks)} blocs disponibles"
        
        demo = gr.Interface(
            fn=hello_dataflowlab,
            inputs=[],
            outputs="text",
            title="DataFlowLab - Test Simple",
            description="Test de fonctionnement de base"
        )
        
        print("🚀 Lancement de DataFlowLab (mode simple)")
        demo.launch(server_port=7860, share=False)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise

if __name__ == "__main__":
    main()
