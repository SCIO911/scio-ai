"""
SCIO Machine Learning Pipeline

AutoML und Modell-Management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union, Callable
import numpy as np

from scio.core.logging import get_logger
from scio.core.utils import generate_id, now_utc

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Unterstützte Modelltypen."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    KNN = "knn"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"


class TaskType(str, Enum):
    """ML-Aufgabentypen."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CLUSTERING = "clustering"


@dataclass
class ModelMetrics:
    """Modell-Bewertungsmetriken."""

    # Regression
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    # Klassifikation
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None

    # Allgemein
    training_time_ms: Optional[float] = None
    n_samples: int = 0
    n_features: int = 0


@dataclass
class TrainedModel:
    """Ein trainiertes Modell."""

    id: str
    model_type: ModelType
    task_type: TaskType
    parameters: dict[str, Any]
    metrics: ModelMetrics
    feature_names: list[str] = field(default_factory=list)
    _weights: Optional[np.ndarray] = None
    _bias: Optional[float] = None
    created_at: datetime = field(default_factory=now_utc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Macht Vorhersagen."""
        if self._weights is None:
            raise ValueError("Model not trained")

        if self.model_type == ModelType.LINEAR_REGRESSION:
            return X @ self._weights + (self._bias or 0)

        elif self.model_type == ModelType.LOGISTIC_REGRESSION:
            z = X @ self._weights + (self._bias or 0)
            probs = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return (probs >= 0.5).astype(int)

        else:
            return X @ self._weights + (self._bias or 0)


class FeatureEngineering:
    """
    Feature Engineering Werkzeuge.

    Features:
    - Normalisierung
    - Standardisierung
    - Polynomiale Features
    - Feature-Auswahl
    """

    def __init__(self):
        self._scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def standardize(
        self,
        X: np.ndarray,
        fit: bool = True,
        name: str = "default",
    ) -> np.ndarray:
        """Z-Score Standardisierung."""
        if fit:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.where(std == 0, 1, std)  # Verhindere Division durch 0
            self._scalers[name] = (mean, std)
        else:
            if name not in self._scalers:
                raise ValueError(f"Scaler '{name}' not fitted")
            mean, std = self._scalers[name]

        return (X - mean) / std

    def normalize(
        self,
        X: np.ndarray,
        fit: bool = True,
        name: str = "default",
    ) -> np.ndarray:
        """Min-Max Normalisierung auf [0, 1]."""
        if fit:
            min_val = np.min(X, axis=0)
            max_val = np.max(X, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            self._scalers[name] = (min_val, range_val)
        else:
            if name not in self._scalers:
                raise ValueError(f"Scaler '{name}' not fitted")
            min_val, range_val = self._scalers[name]

        return (X - min_val) / range_val

    def polynomial_features(
        self,
        X: np.ndarray,
        degree: int = 2,
        include_bias: bool = False,
    ) -> np.ndarray:
        """Erzeugt polynomiale Features."""
        n_samples, n_features = X.shape
        features = [X]

        if include_bias:
            features.insert(0, np.ones((n_samples, 1)))

        for d in range(2, degree + 1):
            for i in range(n_features):
                features.append(X[:, i:i+1] ** d)

        # Interaktionsterme für Grad 2
        if degree >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    features.append(X[:, i:i+1] * X[:, j:j+1])

        return np.hstack(features)

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "correlation",
        k: int = 10,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Feature-Auswahl.

        Args:
            X: Feature-Matrix
            y: Zielvariable
            method: "correlation" oder "variance"
            k: Anzahl auszuwählender Features
        """
        n_features = X.shape[1]
        k = min(k, n_features)

        if method == "correlation":
            # Korrelation mit Zielvariable
            scores = []
            for i in range(n_features):
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                scores.append(corr if not np.isnan(corr) else 0)
            scores = np.array(scores)

        elif method == "variance":
            # Varianz der Features
            scores = np.var(X, axis=0)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Top-k Features
        top_indices = np.argsort(scores)[::-1][:k]
        top_indices = sorted(top_indices)

        return X[:, top_indices], list(top_indices)


class ModelTrainer:
    """
    Modelltraining für verschiedene Algorithmen.
    """

    def __init__(self):
        logger.info("ModelTrainer initialized")

    def train_linear_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularization: float = 0.0,
    ) -> TrainedModel:
        """Trainiert lineare Regression."""
        import time
        start = time.perf_counter()

        n_samples, n_features = X.shape

        # Ridge Regression (L2)
        if regularization > 0:
            I = np.eye(n_features)
            weights = np.linalg.solve(
                X.T @ X + regularization * I,
                X.T @ y
            )
        else:
            # OLS
            weights, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Bias (falls nicht in X enthalten)
        y_pred = X @ weights
        bias = np.mean(y - y_pred)

        elapsed = (time.perf_counter() - start) * 1000

        # Metriken
        y_pred_final = X @ weights + bias
        metrics = self._regression_metrics(y, y_pred_final)
        metrics.training_time_ms = elapsed
        metrics.n_samples = n_samples
        metrics.n_features = n_features

        return TrainedModel(
            id=generate_id("model"),
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=TaskType.REGRESSION,
            parameters={"regularization": regularization},
            metrics=metrics,
            _weights=weights,
            _bias=bias,
        )

    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        regularization: float = 0.0,
    ) -> TrainedModel:
        """Trainiert logistische Regression mit Gradient Descent."""
        import time
        start = time.perf_counter()

        n_samples, n_features = X.shape

        # Initialisiere Gewichte
        weights = np.zeros(n_features)
        bias = 0.0

        for iteration in range(max_iterations):
            # Forward Pass
            z = X @ weights + bias
            y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

            # Gradienten
            error = y_pred - y
            d_weights = (X.T @ error) / n_samples + regularization * weights
            d_bias = np.mean(error)

            # Update
            weights -= learning_rate * d_weights
            bias -= learning_rate * d_bias

        elapsed = (time.perf_counter() - start) * 1000

        # Metriken
        z = X @ weights + bias
        y_pred_proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        y_pred = (y_pred_proba >= 0.5).astype(int)
        metrics = self._classification_metrics(y, y_pred, y_pred_proba)
        metrics.training_time_ms = elapsed
        metrics.n_samples = n_samples
        metrics.n_features = n_features

        return TrainedModel(
            id=generate_id("model"),
            model_type=ModelType.LOGISTIC_REGRESSION,
            task_type=TaskType.BINARY_CLASSIFICATION,
            parameters={
                "learning_rate": learning_rate,
                "max_iterations": max_iterations,
                "regularization": regularization,
            },
            metrics=metrics,
            _weights=weights,
            _bias=bias,
        )

    def train_knn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        task: TaskType = TaskType.BINARY_CLASSIFICATION,
    ) -> TrainedModel:
        """Trainiert K-Nearest Neighbors."""
        import time
        start = time.perf_counter()

        n_samples, n_features = X.shape
        elapsed = (time.perf_counter() - start) * 1000

        # KNN speichert nur Trainingsdaten
        model = TrainedModel(
            id=generate_id("model"),
            model_type=ModelType.KNN,
            task_type=task,
            parameters={"k": k},
            metrics=ModelMetrics(
                training_time_ms=elapsed,
                n_samples=n_samples,
                n_features=n_features,
            ),
        )

        # Speichere Trainingsdaten als "Gewichte"
        model._weights = np.hstack([X, y.reshape(-1, 1)])

        return model

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Berechnet Regressionsmetriken."""
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return ModelMetrics(
            mse=mse,
            rmse=float(np.sqrt(mse)),
            mae=mae,
            r2=float(r2),
        )

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> ModelMetrics:
        """Berechnet Klassifikationsmetriken."""
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # AUC-ROC (falls Wahrscheinlichkeiten verfügbar)
        auc = None
        if y_proba is not None:
            auc = self._calculate_auc(y_true, y_proba)

        return ModelMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            auc_roc=auc,
        )

    def _calculate_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Berechnet AUC-ROC."""
        # Sortiere nach Wahrscheinlichkeit
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # Berechne TPR und FPR für verschiedene Schwellenwerte
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_prev, fpr_prev = 0, 0
        auc = 0

        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tpr = (np.sum(y_true_sorted[:i+1])) / n_pos
                fpr = (i + 1 - np.sum(y_true_sorted[:i+1])) / n_neg
                auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
                tpr_prev, fpr_prev = tpr, fpr

        return float(auc)


class ModelEvaluator:
    """Modellbewertung und Vergleich."""

    def cross_validate(
        self,
        trainer: ModelTrainer,
        X: np.ndarray,
        y: np.ndarray,
        k_folds: int = 5,
        model_type: ModelType = ModelType.LINEAR_REGRESSION,
    ) -> dict[str, Any]:
        """K-Fold Cross-Validation."""
        n_samples = len(X)
        fold_size = n_samples // k_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        metrics_list = []

        for fold in range(k_folds):
            # Split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_samples
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # Train
            if model_type == ModelType.LINEAR_REGRESSION:
                model = trainer.train_linear_regression(X_train, y_train)
            elif model_type == ModelType.LOGISTIC_REGRESSION:
                model = trainer.train_logistic_regression(X_train, y_train)
            else:
                model = trainer.train_linear_regression(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)

            if model.task_type == TaskType.REGRESSION:
                fold_metrics = trainer._regression_metrics(y_val, y_pred)
            else:
                fold_metrics = trainer._classification_metrics(y_val, (y_pred >= 0.5).astype(int))

            metrics_list.append(fold_metrics)

        # Aggregiere
        result = {"k_folds": k_folds}

        if metrics_list[0].mse is not None:
            result["mse_mean"] = np.mean([m.mse for m in metrics_list])
            result["mse_std"] = np.std([m.mse for m in metrics_list])
            result["r2_mean"] = np.mean([m.r2 for m in metrics_list])
            result["r2_std"] = np.std([m.r2 for m in metrics_list])

        if metrics_list[0].accuracy is not None:
            result["accuracy_mean"] = np.mean([m.accuracy for m in metrics_list])
            result["accuracy_std"] = np.std([m.accuracy for m in metrics_list])
            result["f1_mean"] = np.mean([m.f1 for m in metrics_list])
            result["f1_std"] = np.std([m.f1 for m in metrics_list])

        return result


class AutoMLPipeline:
    """
    AutoML Pipeline für automatisiertes Machine Learning.

    Features:
    - Automatische Datenvorverarbeitung
    - Modellauswahl
    - Hyperparameter-Tuning
    - Ensemble-Methoden
    """

    def __init__(self):
        self.feature_eng = FeatureEngineering()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        logger.info("AutoMLPipeline initialized")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: Optional[TaskType] = None,
        max_models: int = 5,
        cv_folds: int = 3,
    ) -> dict[str, Any]:
        """
        Findet automatisch das beste Modell.

        Args:
            X: Feature-Matrix
            y: Zielvariable
            task: Aufgabentyp (wird automatisch erkannt wenn None)
            max_models: Maximale Anzahl zu testender Modelle
            cv_folds: Anzahl Cross-Validation Folds
        """
        # Aufgabentyp erkennen
        if task is None:
            unique_values = np.unique(y)
            if len(unique_values) <= 10 and np.all(unique_values == unique_values.astype(int)):
                if len(unique_values) == 2:
                    task = TaskType.BINARY_CLASSIFICATION
                else:
                    task = TaskType.MULTICLASS_CLASSIFICATION
            else:
                task = TaskType.REGRESSION

        # Datenvorverarbeitung
        X_processed = self.feature_eng.standardize(X.copy())

        # Modelle zum Testen
        models_to_try = []

        if task == TaskType.REGRESSION:
            models_to_try = [
                (ModelType.LINEAR_REGRESSION, {}),
                (ModelType.LINEAR_REGRESSION, {"regularization": 0.1}),
                (ModelType.LINEAR_REGRESSION, {"regularization": 1.0}),
            ]
        else:
            models_to_try = [
                (ModelType.LOGISTIC_REGRESSION, {"learning_rate": 0.01}),
                (ModelType.LOGISTIC_REGRESSION, {"learning_rate": 0.1}),
                (ModelType.LOGISTIC_REGRESSION, {"regularization": 0.1}),
            ]

        # Trainiere und evaluiere Modelle
        results = []
        for model_type, params in models_to_try[:max_models]:
            try:
                cv_result = self.evaluator.cross_validate(
                    self.trainer, X_processed, y, cv_folds, model_type
                )
                cv_result["model_type"] = model_type.value
                cv_result["parameters"] = params
                results.append(cv_result)
            except Exception as e:
                logger.warning(f"Model {model_type} failed: {e}")

        # Wähle bestes Modell
        if task == TaskType.REGRESSION:
            best = min(results, key=lambda r: r.get("mse_mean", float("inf")))
        else:
            best = max(results, key=lambda r: r.get("accuracy_mean", 0))

        # Trainiere finales Modell auf allen Daten
        best_type = ModelType(best["model_type"])
        best_params = best["parameters"]

        if best_type == ModelType.LINEAR_REGRESSION:
            final_model = self.trainer.train_linear_regression(
                X_processed, y, **best_params
            )
        else:
            final_model = self.trainer.train_logistic_regression(
                X_processed, y, **best_params
            )

        return {
            "best_model": final_model,
            "task_type": task.value,
            "cv_results": results,
            "best_cv_result": best,
        }
