from typing import Dict, Type, List
from .block_base import BlockBase
import logging

logger = logging.getLogger(__name__)

class BlockRegistry:
    """
    Registre centralisé pour tous les blocs disponibles dans DataFlowLab.
    Auto-découverte et enregistrement des blocs.
    """
    _registry: Dict[str, Type[BlockBase]] = {}
    _categories: Dict[str, List[str]] = {}

    @classmethod
    def register(cls, name: str, block_cls: Type[BlockBase], category: str = "other") -> None:
        """Enregistre un bloc dans le registre."""
        try:
            cls._registry[name] = block_cls
            if category not in cls._categories:
                cls._categories[category] = []
            cls._categories[category].append(name)
            logger.info(f"Bloc {name} enregistré dans la catégorie {category}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du bloc {name}: {e}")

    @classmethod
    def get_block(cls, name: str) -> Type[BlockBase]:
        """Récupère une classe de bloc par son nom."""
        if name not in cls._registry:
            raise ValueError(f"Bloc '{name}' non trouvé dans le registre")
        return cls._registry[name]

    @classmethod
    def list_blocks(cls) -> Dict[str, Type[BlockBase]]:
        """Liste tous les blocs enregistrés."""
        return dict(cls._registry)
    
    @classmethod
    def list_by_category(cls) -> Dict[str, List[str]]:
        """Liste les blocs par catégorie."""
        return dict(cls._categories)
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Retourne la liste des catégories."""
        return list(cls._categories.keys())
    
    @classmethod
    def auto_discover_blocks(cls) -> None:
        """Auto-découverte des blocs dans les modules."""
        try:
            # Import data_input
            import dataflowlab.blocks.data_input.csv_loader
            import dataflowlab.blocks.data_input.excel_loader
            import dataflowlab.blocks.data_input.json_loader
            import dataflowlab.blocks.data_input.sql_connector
            import dataflowlab.blocks.data_input.data_exporter
            import dataflowlab.blocks.data_input.dataset_splitter
            
            # Import data_cleaning
            import dataflowlab.blocks.data_cleaning.missing_values_handler
            import dataflowlab.blocks.data_cleaning.duplicate_remover
            import dataflowlab.blocks.data_cleaning.data_type_converter
            import dataflowlab.blocks.data_cleaning.column_renamer
            import dataflowlab.blocks.data_cleaning.row_filter
            
            # Import feature_engineering - TOUS
            import dataflowlab.blocks.feature_engineering.feature_scaler
            import dataflowlab.blocks.feature_engineering.one_hot_encoder
            import dataflowlab.blocks.feature_engineering.label_encoder
            import dataflowlab.blocks.feature_engineering.pca_transformer
            import dataflowlab.blocks.feature_engineering.target_encoder
            import dataflowlab.blocks.feature_engineering.polynomial_features
            import dataflowlab.blocks.feature_engineering.feature_selector
            import dataflowlab.blocks.feature_engineering.feature_interactions
            import dataflowlab.blocks.feature_engineering.binning_transformer
            import dataflowlab.blocks.feature_engineering.date_feature_extractor
            
            # Import supervised - TOUS
            import dataflowlab.blocks.supervised.linear_regression
            import dataflowlab.blocks.supervised.logistic_regression
            import dataflowlab.blocks.supervised.random_forest
            import dataflowlab.blocks.supervised.decision_tree
            import dataflowlab.blocks.supervised.knn
            import dataflowlab.blocks.supervised.gradient_boosting
            import dataflowlab.blocks.supervised.naive_bayes
            import dataflowlab.blocks.supervised.neural_network
            import dataflowlab.blocks.supervised.svm
            import dataflowlab.blocks.supervised.regularized_regression
            
            # Import unsupervised - TOUS
            import dataflowlab.blocks.unsupervised.kmeans_clustering
            import dataflowlab.blocks.unsupervised.dbscan_clustering
            import dataflowlab.blocks.unsupervised.hierarchical_clustering
            import dataflowlab.blocks.unsupervised.gaussian_mixture
            import dataflowlab.blocks.unsupervised.anomaly_detection
            import dataflowlab.blocks.unsupervised.association_rules
            
            # Import evaluation - TOUS
            import dataflowlab.blocks.evaluation.classification_metrics
            import dataflowlab.blocks.evaluation.regression_metrics
            import dataflowlab.blocks.evaluation.cross_validation
            import dataflowlab.blocks.evaluation.correlation_analysis
            import dataflowlab.blocks.evaluation.confusion_matrix
            import dataflowlab.blocks.evaluation.feature_importance
            import dataflowlab.blocks.evaluation.learning_curves
            import dataflowlab.blocks.evaluation.prediction_visualizer
            import dataflowlab.blocks.evaluation.shap_explainer
            
            # Import timeseries - TOUS
            import dataflowlab.blocks.timeseries.time_series_decomposition
            import dataflowlab.blocks.timeseries.lag_features
            import dataflowlab.blocks.timeseries.arima_model
            import dataflowlab.blocks.timeseries.rolling_statistics
            import dataflowlab.blocks.timeseries.seasonality_decomposer
            import dataflowlab.blocks.timeseries.time_series_cross_validator
            import dataflowlab.blocks.timeseries.time_series_loader
            
            # Import advanced - TOUS
            import dataflowlab.blocks.advanced.tfidf_vectorizer
            import dataflowlab.blocks.advanced.model_ensembler
            import dataflowlab.blocks.advanced.custom_code_block
            import dataflowlab.blocks.advanced.image_loader
            import dataflowlab.blocks.advanced.text_preprocessor
            import dataflowlab.blocks.advanced.pipeline_combiner
            
            logger.info(f"Auto-découverte terminée: {len(cls._registry)} blocs enregistrés")
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'auto-découverte: {e}")

    def __init__(self):
        """Initialise le registre et déclenche l'auto-découverte."""
        self.auto_discover_blocks()
    
    @property
    def blocks(self) -> Dict[str, Type[BlockBase]]:
        """Accès aux blocs enregistrés pour compatibilité."""
        return self._registry
    
    @property
    def categories(self) -> Dict[str, List[str]]:
        """Accès aux catégories pour compatibilité."""
        return self._categories

    @classmethod 
    def register_block(cls, name: str, block_cls: Type[BlockBase]) -> None:
        """Méthode simplifiée pour l'enregistrement automatique."""
        # Extraire la catégorie du bloc si disponible
        category = 'other'
        try:
            # Essayer d'instancier temporairement pour récupérer la catégorie
            dummy = block_cls(params={})
            if hasattr(dummy, 'category'):
                category = dummy.category
            elif hasattr(dummy, '_category'):
                category = dummy._category
        except Exception as e:
            # Si l'instanciation échoue, essayer d'analyser le code source ou les attributs de classe
            try:
                # Vérifier si la catégorie est définie comme attribut de classe
                if hasattr(block_cls, '_category'):
                    category = block_cls._category
                elif hasattr(block_cls, 'category'):
                    category = block_cls.category
                else:
                    # Analyser le nom du module pour deviner la catégorie
                    module_name = block_cls.__module__
                    if 'data_input' in module_name:
                        category = 'data_input'
                    elif 'data_cleaning' in module_name:
                        category = 'data_cleaning'
                    elif 'feature_engineering' in module_name:
                        category = 'feature_engineering'
                    elif 'supervised' in module_name:
                        category = 'supervised'
                    elif 'unsupervised' in module_name:
                        category = 'unsupervised'
                    elif 'evaluation' in module_name:
                        category = 'evaluation'
                    elif 'advanced' in module_name:
                        category = 'advanced'
                    elif 'timeseries' in module_name:
                        category = 'timeseries'
            except Exception:
                pass
        
        cls.register(name, block_cls, category)
