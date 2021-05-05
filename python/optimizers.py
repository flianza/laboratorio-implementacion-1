import optuna
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import abc
from datatable import Frame
from typing import Tuple, TypeVar, Generic, Union
from models import LightGBMModel, StudyModel, XGBoostModel
from utils import get_score_corte

TStudyModel = TypeVar('TStudyModel', bound=StudyModel)


class ModelOptimizer(abc.ABC, Generic[TStudyModel]):
    def __init__(self, X: Frame, y: np.ndarray, weights: np.ndarray,
                 X_val: Frame, y_val: np.ndarray, weights_val: np.ndarray,
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
        if n > len(self.models):
            n = len(self.models)
        return sorted(self.models, key=lambda m: m.get_score(), reverse=True)[:n]

    @staticmethod
    def _evaluar_funcion_ganancia(scores, labels, weights, score_corte) -> Tuple[str, int]:
        scores_prob_corte = scores > score_corte

        if len(weights) == 0 or weights is None:
            ganancias = np.where(labels == 1, 29250, -750)
        else:
            ganancias = np.where((labels == 1) & (weights > 1), 29250, -750)

        ganancia_actual = np.dot(scores_prob_corte, ganancias)

        return 'ganancia', ganancia_actual


class LightGBMOptimizer(ModelOptimizer[LightGBMModel]):
    def __init__(self, X: Frame, y: Frame, weights: Union[Frame, None],
                 X_val: Frame, y_val: Frame, weights_val: Union[Frame, None],
                 prob_corte_min=0.02, prob_corte_max=0.03, prob_corte=0.025):
        super().__init__(X, y.to_numpy(), weights.to_numpy() if weights is not None else None,
                         X_val, y_val.to_numpy(), weights_val.to_numpy() if weights_val is not None else None, prob_corte)
        self.prob_corte_min = prob_corte_min
        self.prob_corte_max = prob_corte_max

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
        score_corte = self.prob_corte

        nombre, valor = self._evaluar_funcion_ganancia(scores, labels, weights, score_corte)

        return nombre, valor, True

    def evaluate_trial(self, trial: optuna.Trial) -> float:
        self.prob_corte = trial.suggest_float('prob_corte', self.prob_corte_min, self.prob_corte_max)

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
                            lgb.Dataset(self.X, label=self.y, weight=self.weights, feature_name=self.X.names),
                            num_boost_round=99999,
                            feval=self._evaluate,
                            early_stopping_rounds=int(50 + 5 / params['learning_rate']),
                            valid_sets=[lgb.Dataset(self.X_val, label=self.y_val, weight=self.weights_val)],
                            valid_names=['validation'],
                            verbose_eval=False)

        self.models.append(LightGBMModel(booster))

        return booster.best_score['validation']['ganancia']


class XGBoostOptimizer(ModelOptimizer[XGBoostModel]):
    def __init__(self, X: Frame, y: np.ndarray, weights: np.ndarray,
                 X_val: Frame, y_val: np.ndarray, weights_val: np.ndarray,
                 prob_corte_min=0.02, prob_corte_max=0.03, prob_corte=0.025):
        super().__init__(X, y, weights, X_val, y_val, weights_val, prob_corte)
        self.__actualizar_prob_corte(prob_corte)
        self.prob_corte_min = prob_corte_min
        self.prob_corte_max = prob_corte_max

        self.fixed_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'grow_policy': 'lossguide'
        }

    def __actualizar_prob_corte(self, prob_corte):
        self.prob_corte = prob_corte
        self.score_corte = get_score_corte(prob_corte)

    def _evaluate(self, scores: np.ndarray, clases: xgb.DMatrix) -> Tuple[str, int]:
        labels = clases.get_label()
        weights = clases.get_weight()
        score_corte = self.score_corte

        return self._evaluar_funcion_ganancia(scores, labels, weights, score_corte)

    def evaluate_trial(self, trial: optuna.Trial) -> float:
        self.__actualizar_prob_corte(trial.suggest_float('prob_corte', self.prob_corte_min, self.prob_corte_max))

        variable_params = {
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 100),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_float('min_child_weight', 0, 5),
            'max_leaves': trial.suggest_int('max_leaves', 32, 512),
            'max_depth': trial.suggest_int('max_depth', 0, 50),
            'max_bin': trial.suggest_int('max_bin', 20, 40)
        }

        params = {**self.fixed_params, **variable_params}

        booster = xgb.train(params,
                            xgb.DMatrix(self.X, label=self.y, weight=self.weights),
                            num_boost_round=99999,
                            feval=self._evaluate,
                            maximize=True,
                            early_stopping_rounds=int(50 + 5 / params['eta']),
                            evals=[(xgb.DMatrix(self.X_val, label=self.y_val, weight=self.weights_val), 'validation')],
                            verbose_eval=False)

        self.models.append(XGBoostModel(booster))

        return booster.best_score
