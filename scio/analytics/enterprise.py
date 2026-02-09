"""
SCIO Enterprise Analytics (MEGA-UPGRADE)

Big Data Processing und Enterprise-Grade Analytics.

Features:
- Dask Integration für Big Data
- Polars für schnelle DataFrames
- XGBoost/LightGBM/CatBoost Models
- SHAP Explainability
- Feature Importance Analysis
- Distributed Computing Support
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union, List, Dict, Callable
import numpy as np

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)

# Optional Big Data Libraries
DASK_AVAILABLE = False
try:
    import dask.dataframe as dd
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    pass

POLARS_AVAILABLE = False
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pass

XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

CATBOOST_AVAILABLE = False
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

OPTUNA_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    pass


class BoostingFramework(str, Enum):
    """Unterstützte Boosting Frameworks."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


@dataclass
class EnterpriseModelMetrics:
    """Enterprise Model Metriken."""
    # Performance
    accuracy: Optional[float] = None
    auc_roc: Optional[float] = None
    f1_score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    # Training
    training_time_seconds: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    n_iterations: int = 0

    # Feature Importance
    top_features: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-Validation
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0


class BigDataProcessor:
    """
    MEGA-UPGRADE: Big Data Processing mit Dask und Polars

    Features:
    - Lazy Evaluation für große Datasets
    - Parallele Verarbeitung
    - Speichereffiziente Operationen
    - Support für Parquet, CSV, JSON
    """

    def __init__(self, use_dask: bool = True, n_workers: int = None):
        self._use_dask = use_dask and DASK_AVAILABLE
        self._use_polars = POLARS_AVAILABLE
        self._dask_client = None
        self._n_workers = n_workers

        if self._use_dask:
            try:
                self._dask_client = Client(n_workers=n_workers, silence_logs=40)
                logger.info(f"Dask Client gestartet: {self._dask_client.dashboard_link}")
            except Exception as e:
                logger.warning(f"Dask Client Start fehlgeschlagen: {e}")
                self._use_dask = False

    def load_large_csv(
        self,
        path: str,
        chunk_size: int = 100_000,
        columns: List[str] = None,
    ) -> Any:
        """
        Lädt große CSV-Datei effizient.

        Args:
            path: Pfad zur CSV-Datei
            chunk_size: Zeilen pro Chunk
            columns: Optionale Spaltenauswahl

        Returns:
            Dask DataFrame oder Polars LazyFrame
        """
        if self._use_dask:
            df = dd.read_csv(path, usecols=columns, blocksize=f"{chunk_size * 100}B")
            logger.info(f"Dask DataFrame geladen: {path}")
            return df

        elif self._use_polars:
            lf = pl.scan_csv(path, n_rows=None)
            if columns:
                lf = lf.select(columns)
            logger.info(f"Polars LazyFrame geladen: {path}")
            return lf

        else:
            raise RuntimeError("Keine Big Data Library verfügbar")

    def load_parquet(
        self,
        path: str,
        columns: List[str] = None,
    ) -> Any:
        """Lädt Parquet-Datei."""
        if self._use_dask:
            df = dd.read_parquet(path, columns=columns)
            return df

        elif self._use_polars:
            lf = pl.scan_parquet(path)
            if columns:
                lf = lf.select(columns)
            return lf

        else:
            raise RuntimeError("Keine Big Data Library verfügbar")

    def aggregate(
        self,
        df: Any,
        group_by: List[str],
        aggregations: Dict[str, str],
    ) -> Any:
        """
        Aggregiert DataFrame.

        Args:
            df: DataFrame (Dask oder Polars)
            group_by: Gruppierungs-Spalten
            aggregations: {column: 'sum'|'mean'|'count'|'min'|'max'}

        Returns:
            Aggregiertes DataFrame
        """
        if hasattr(df, 'groupby'):  # Dask
            agg_dict = {}
            for col, agg in aggregations.items():
                agg_dict[col] = agg
            result = df.groupby(group_by).agg(agg_dict)
            return result.compute()

        elif hasattr(df, 'group_by'):  # Polars
            agg_exprs = []
            for col, agg in aggregations.items():
                if agg == 'sum':
                    agg_exprs.append(pl.col(col).sum())
                elif agg == 'mean':
                    agg_exprs.append(pl.col(col).mean())
                elif agg == 'count':
                    agg_exprs.append(pl.col(col).count())
                elif agg == 'min':
                    agg_exprs.append(pl.col(col).min())
                elif agg == 'max':
                    agg_exprs.append(pl.col(col).max())
            return df.group_by(group_by).agg(agg_exprs).collect()

        else:
            raise ValueError("Unbekannter DataFrame-Typ")

    def to_numpy(self, df: Any) -> np.ndarray:
        """Konvertiert zu NumPy Array."""
        if hasattr(df, 'compute'):  # Dask
            return df.compute().values
        elif hasattr(df, 'collect'):  # Polars LazyFrame
            return df.collect().to_numpy()
        elif hasattr(df, 'to_numpy'):  # Polars DataFrame
            return df.to_numpy()
        else:
            return np.array(df)

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück."""
        return {
            'dask_available': DASK_AVAILABLE,
            'polars_available': POLARS_AVAILABLE,
            'using_dask': self._use_dask,
            'dask_dashboard': self._dask_client.dashboard_link if self._dask_client else None,
        }

    def shutdown(self):
        """Beendet Dask Client."""
        if self._dask_client:
            self._dask_client.close()


class GradientBoostingTrainer:
    """
    MEGA-UPGRADE: Enterprise Gradient Boosting

    Unterstützt:
    - XGBoost
    - LightGBM
    - CatBoost

    Features:
    - Automatische Hyperparameter-Optimierung
    - Cross-Validation
    - Feature Importance
    - SHAP Explainability
    """

    def __init__(self, framework: BoostingFramework = BoostingFramework.XGBOOST):
        self.framework = framework
        self._model = None
        self._feature_names = None
        self._shap_explainer = None

        # Prüfe Verfügbarkeit
        if framework == BoostingFramework.XGBOOST and not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost nicht installiert")
        if framework == BoostingFramework.LIGHTGBM and not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM nicht installiert")
        if framework == BoostingFramework.CATBOOST and not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost nicht installiert")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
        params: Dict[str, Any] = None,
        n_estimators: int = 100,
        feature_names: List[str] = None,
    ) -> EnterpriseModelMetrics:
        """
        Trainiert Gradient Boosting Modell.

        Args:
            X: Feature Matrix
            y: Labels
            task: 'classification' oder 'regression'
            params: Modell-Parameter
            n_estimators: Anzahl Bäume
            feature_names: Feature-Namen

        Returns:
            EnterpriseModelMetrics
        """
        import time
        start_time = time.time()

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        params = params or {}

        if self.framework == BoostingFramework.XGBOOST:
            if task == "classification":
                self._model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    **params
                )
            else:
                self._model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    **params
                )
            self._model.fit(X, y)

        elif self.framework == BoostingFramework.LIGHTGBM:
            if task == "classification":
                self._model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    **params
                )
            else:
                self._model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    **params
                )
            self._model.fit(X, y)

        elif self.framework == BoostingFramework.CATBOOST:
            if task == "classification":
                self._model = cb.CatBoostClassifier(
                    iterations=n_estimators,
                    verbose=False,
                    **params
                )
            else:
                self._model = cb.CatBoostRegressor(
                    iterations=n_estimators,
                    verbose=False,
                    **params
                )
            self._model.fit(X, y)

        training_time = time.time() - start_time

        # Feature Importance
        importance = self.get_feature_importance()

        metrics = EnterpriseModelMetrics(
            training_time_seconds=training_time,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_iterations=n_estimators,
            top_features=importance[:10],
        )

        # Metriken berechnen
        y_pred = self._model.predict(X)

        if task == "classification":
            metrics.accuracy = float(np.mean(y_pred == y))
        else:
            metrics.rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
            metrics.mae = float(np.mean(np.abs(y - y_pred)))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics.r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        logger.info(f"Training abgeschlossen: {self.framework.value} in {training_time:.2f}s")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Macht Vorhersagen."""
        if self._model is None:
            raise ValueError("Modell nicht trainiert")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Gibt Wahrscheinlichkeiten zurück (nur für Classification)."""
        if self._model is None:
            raise ValueError("Modell nicht trainiert")
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X)
        raise ValueError("Modell unterstützt keine Wahrscheinlichkeiten")

    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Gibt Feature Importance zurück."""
        if self._model is None:
            return []

        importance = self._model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        return [
            {
                'feature': self._feature_names[i],
                'importance': float(importance[i]),
                'rank': rank + 1,
            }
            for rank, i in enumerate(sorted_idx)
        ]

    def explain_with_shap(self, X: np.ndarray, max_samples: int = 1000) -> Dict[str, Any]:
        """
        MEGA-UPGRADE: SHAP Explainability

        Args:
            X: Feature Matrix
            max_samples: Max Samples für Erklärung

        Returns:
            dict mit shap_values und summary
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP nicht installiert'}

        if self._model is None:
            return {'error': 'Modell nicht trainiert'}

        # Subsample wenn nötig
        if X.shape[0] > max_samples:
            idx = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        # Erstelle Explainer
        if self._shap_explainer is None:
            self._shap_explainer = shap.Explainer(self._model)

        shap_values = self._shap_explainer(X_sample)

        # Berechne durchschnittliche Importance
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        feature_importance = [
            {
                'feature': self._feature_names[i],
                'mean_abs_shap': float(mean_abs_shap[i]),
                'rank': rank + 1,
            }
            for rank, i in enumerate(sorted_idx[:20])
        ]

        return {
            'feature_importance': feature_importance,
            'n_samples_explained': X_sample.shape[0],
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        task: str = "classification",
    ) -> Dict[str, Any]:
        """
        Cross-Validation.

        Returns:
            dict mit scores und Statistiken
        """
        from sklearn.model_selection import cross_val_score

        scoring = 'accuracy' if task == 'classification' else 'neg_mean_squared_error'

        scores = cross_val_score(
            self._model,
            X, y,
            cv=n_folds,
            scoring=scoring
        )

        if task == 'regression':
            scores = -scores  # Negatives MSE zu positivem umwandeln

        return {
            'scores': scores.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'n_folds': n_folds,
        }


class HyperparameterOptimizer:
    """
    MEGA-UPGRADE: AutoML mit Optuna

    Features:
    - Bayesian Optimization
    - Pruning von schlechten Trials
    - Parallele Suche
    """

    def __init__(self, n_trials: int = 100, n_jobs: int = -1):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna nicht installiert")

        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self._study = None
        self._best_params = None

    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
        cv_folds: int = 3,
    ) -> Dict[str, Any]:
        """
        Optimiert XGBoost Hyperparameter.

        Returns:
            dict mit best_params, best_score, n_trials
        """
        if not XGBOOST_AVAILABLE:
            return {'error': 'XGBoost nicht installiert'}

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }

            if task == "classification":
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            else:
                model = xgb.XGBRegressor(**params)

            from sklearn.model_selection import cross_val_score
            scoring = 'accuracy' if task == 'classification' else 'neg_mean_squared_error'

            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            return np.mean(scores)

        self._study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(),
        )

        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        self._best_params = self._study.best_params

        return {
            'best_params': self._best_params,
            'best_score': self._study.best_value,
            'n_trials': len(self._study.trials),
        }

    def get_best_params(self) -> Dict[str, Any]:
        """Gibt beste Parameter zurück."""
        return self._best_params or {}


class AnomalyDetector:
    """
    MEGA-UPGRADE: Anomaly Detection

    Algorithmen:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - One-Class SVM
    """

    def __init__(self, algorithm: str = "isolation_forest"):
        self.algorithm = algorithm
        self._model = None

    def fit(
        self,
        X: np.ndarray,
        contamination: float = 0.1,
    ):
        """
        Trainiert Anomaly Detector.

        Args:
            X: Feature Matrix
            contamination: Erwarteter Anteil Anomalien
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM

        if self.algorithm == "isolation_forest":
            self._model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )
        elif self.algorithm == "lof":
            self._model = LocalOutlierFactor(
                contamination=contamination,
                n_jobs=-1,
                novelty=True,
            )
        elif self.algorithm == "one_class_svm":
            self._model = OneClassSVM(nu=contamination)
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algorithm}")

        self._model.fit(X)
        logger.info(f"Anomaly Detector trainiert: {self.algorithm}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Erkennt Anomalien.

        Returns:
            Array mit -1 für Anomalien, 1 für normal
        """
        if self._model is None:
            raise ValueError("Modell nicht trainiert")

        return self._model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Gibt Anomalie-Scores zurück.

        Niedrigere Scores = mehr anomal
        """
        if self._model is None:
            raise ValueError("Modell nicht trainiert")

        if hasattr(self._model, 'score_samples'):
            return self._model.score_samples(X)
        elif hasattr(self._model, 'decision_function'):
            return self._model.decision_function(X)
        else:
            # Fallback
            predictions = self._model.predict(X)
            return predictions.astype(float)


# Factory Functions
def get_big_data_processor(**kwargs) -> BigDataProcessor:
    """Erstellt BigDataProcessor."""
    return BigDataProcessor(**kwargs)


def get_gradient_boosting_trainer(framework: str = "xgboost") -> GradientBoostingTrainer:
    """Erstellt GradientBoostingTrainer."""
    return GradientBoostingTrainer(BoostingFramework(framework))


def get_hyperparameter_optimizer(**kwargs) -> HyperparameterOptimizer:
    """Erstellt HyperparameterOptimizer."""
    return HyperparameterOptimizer(**kwargs)


def get_anomaly_detector(algorithm: str = "isolation_forest") -> AnomalyDetector:
    """Erstellt AnomalyDetector."""
    return AnomalyDetector(algorithm)
