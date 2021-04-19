import optuna
import numpy as np
import lightgbm as lgb
import datatable as dt
import abc
from typing import Tuple, TypeVar, Generic

from models import LightGbmModel, StudyModel

TStudyModel = TypeVar('TStudyModel', bound=StudyModel)


class ModelOptimizer(abc.ABC, Generic[TStudyModel]):
    def __init__(self, X: dt.Frame, y: np.ndarray, weights: np.ndarray,
                 X_val: dt.Frame, y_val: np.ndarray, weights_val: np.ndarray,
                 prob_corte=0.025):
        self.X = X
        self.y = y
        self.weights = weights
        self.X_val = X_val
        self.y_val = y_val
        self.weights_val = weights_val
        self.prob_corte = prob_corte

        self.models = []

    @abc.abstractmethod
    def evaluate_trial(self, trial: optuna.Trial) -> float:
        pass

    def get_best_models(self, n=5) -> [TStudyModel]:
        return sorted(self.models, key=lambda m: m.get_score(), reverse=True)[:n]


class LightGbmOptimizer(ModelOptimizer[LightGbmModel]):
    def __init__(self, X: dt.Frame, y: np.ndarray, weights: np.ndarray,
                 X_val: dt.Frame, y_val: np.ndarray, weights_val: np.ndarray,
                 prob_corte=0.025):
        super().__init__(X, y, weights, X_val, y_val, weights_val, prob_corte)

        self.fixed_params = {
            'objective': 'binary',
            'metric': 'custom',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'verbose': -1
        }

    def _evaluate(self, scores: np.ndarray, clases: lgb.Dataset) -> Tuple[str, int, bool]:
        labels = clases.get_label()
        weights = clases.get_weight()

        scores_prob_corte = scores > self.prob_corte

        if weights is not None:
            ganancias = np.where((labels == 1) & (weights > 1), 29250, -750)
        else:
            ganancias = np.where(labels == 1, 29250, -750)

        ganancia_actual = np.dot(scores_prob_corte, ganancias)

        return 'ganancia', ganancia_actual, True

    def evaluate_trial(self, trial: optuna.Trial) -> float:
        self.prob_corte = trial.suggest_float('prob_corte', 0.025, 0.05)

        variable_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 5000),
            'num_leaves': trial.suggest_int('num_leaves', 16, 1024),
            'max_bin': trial.suggest_int('max_bin', 15, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 100),
            'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False])
        }

        params = {**self.fixed_params, **variable_params}

        booster = lgb.train(params,
                            lgb.Dataset(self.X, label=self.y, weight=self.weights),
                            num_boost_round=99999,
                            feval=self._evaluate,
                            early_stopping_rounds=int(50 + 5 / params['learning_rate']),
                            valid_sets=[lgb.Dataset(self.X_val, label=self.y_val, weight=self.weights_val)],
                            valid_names=['validation'],
                            verbose_eval=False)

        self.models.append(LightGbmModel(booster, str(len(self.models) + 1)))

        return booster.best_score['validation']['ganancia']
