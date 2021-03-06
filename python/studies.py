import datatable as dt
import optuna
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
from typing import TypeVar, Generic

from optuna.samplers import TPESampler

from optimizers import ModelOptimizer
from utils import start_experiment

TModelOptimizer = TypeVar('TModelOptimizer', bound=ModelOptimizer)


class Study(Generic[TModelOptimizer]):
    def __init__(self, train_params):
        self.train_params = train_params

        self.experiment_number = None
        self.experiment_files_prefix = None
        self.optuna_study = None

    def __enter__(self):
        self.experiment_number, self.experiment_files_prefix = start_experiment(self.train_params)

        storage = f'sqlite:///{self.experiment_files_prefix}_study.db'
        self.optuna_study = optuna.create_study(study_name=self.experiment_number, direction='maximize',
                                                storage=storage, sampler=TPESampler(seed=self.train_params['seed']))

        self.optuna_study.enqueue_trial({
            'prob_corte': 0.1436483860126172,
            'eta': 0.0704510499973727,
            'colsample_bytree': 0.5085641821288974,
            'reg_alpha': 3.929925432157357,
            'reg_lambda': 1.898712607757858,
            'gamma': 1.0978847764856066,
            'min_child_weight': 3.3109497018230534,
            'max_leaves': 114,
            'max_depth': 16,
            'max_bin': 35
        })

        return self

    def __exit__(self, type, value, traceback):
        neptune.stop()

    def optimize(self, optimizer: TModelOptimizer):
        optuna.logging.enable_default_handler()
        self.optuna_study.optimize(optimizer.evaluate_trial, n_trials=self.train_params['trials'],
                                   callbacks=[opt_utils.NeptuneCallback(log_study=True, log_charts=True)])
        opt_utils.log_study_info(self.optuna_study)

        study_importance = optuna.importance.get_param_importances(self.optuna_study)
        study_importance = dt.Frame(variable=list(study_importance.keys()),
                                    valor=list(study_importance.values()))
        return study_importance

    @staticmethod
    def log_csv(data: dt.Frame, filename: str):
        data.to_csv(filename)
        neptune.log_artifact(filename)

    @property
    def best_params(self):
        return self.optuna_study.best_params
