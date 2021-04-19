import datatable as dt
import numpy as np
import xgboost as xgb
import neptune
import optuna
import neptunecontrib.monitoring.optuna as opt_utils
import pandas as pd
from typing import Tuple
from neptunecontrib.monitoring.xgboost import neptune_callback
from optuna.samplers import TPESampler

from utils import calcular_score_corte, crear_experimento

SCRIPT_PARAMS = {
    'seed': 42,
    'trials': 100,
    'e_stop': 100,
    'n_folds': 5,
    'archivo_train': 'datasets/201905_fechas_drifting_montos.csv',
    'archivo_test': 'datasets/201907_fechas_drifting_montos.csv'
}

api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZTBhNWNhZS0xNTNkLTQ2NjktODI0ZC1kOTAyMzhmNzllNDAifQ=='
run = neptune.init('flianza/laboratorio-1-mes', api_token=api_token)
neptune.create_experiment('xgboost-tuning', tags=['xgboost', 'tuning', 'optuna'], params=SCRIPT_PARAMS)
experimento, ubicacion_datos_experimento = crear_experimento(neptune.get_experiment().id)

dataset = dt.fread(SCRIPT_PARAMS['archivo_train'])
dataset['target'] = dataset[:, dt.f.clase_ternaria != 'CONTINUA']
X = dataset[:, dt.f[:].remove([dt.f.numero_de_cliente, dt.f.clase_ternaria, dt.f.target])]
y = dataset[:, 'target']

all_nrounds = {}
score_corte = calcular_score_corte()
np.random.seed(SCRIPT_PARAMS['seed'])


def eval_ganancia(scores: np.ndarray, clases: xgb.DMatrix) -> Tuple[str, float]:
    global score_corte
    y = clases.get_label()
    pesos = clases.get_weight()

    scores_prob_corte = scores > score_corte
    ganancias_por_clase = np.where((y == 1) & (pesos > 1), 29250, -750)
    ganancia = np.dot(scores_prob_corte, ganancias_por_clase)

    return 'ganancia', ganancia


def xgb_evaluate(trial: optuna.Trial) -> float:
    global score_corte
    global all_nrounds

    score_corte = calcular_score_corte(trial.suggest_float('prob_corte', 0.025, 0.05))

    params = {'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'tree_method': 'hist',
              'grow_policy': 'lossguide',
              'eta': trial.suggest_float('eta', 0.01, 0.3),
              'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
              'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
              'reg_lambda': trial.suggest_float('reg_lambda', 0, 100),
              'gamma': trial.suggest_float('gamma', 0, 5),
              'min_child_weight': trial.suggest_float('min_child_weight', 0, 5),
              'max_leaves': trial.suggest_int('max_leaves', 32, 1024),
              'max_depth': trial.suggest_int('max_depth', 0, 100),
              'max_bin': trial.suggest_int('max_bin', 20, 40)}

    cv_result = xgb.cv(params,
                       xgb.DMatrix(X, label=y,
                                   weight=dataset[:, dt.ifelse(dt.f.clase_ternaria == 'BAJA+2', 1.0001, 1)]),
                       num_boost_round=5000,
                       nfold=SCRIPT_PARAMS['n_folds'],
                       stratified=True,
                       feval=eval_ganancia,
                       maximize=True,
                       early_stopping_rounds=SCRIPT_PARAMS['e_stop'],
                       seed=SCRIPT_PARAMS['seed'])

    best_nrounds = cv_result['test-ganancia-mean'].argmax()
    ganancia = cv_result['test-ganancia-mean'].iloc[best_nrounds]
    ganancia *= SCRIPT_PARAMS['n_folds']

    all_nrounds[ganancia] = best_nrounds + 1

    return ganancia


optuna_neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)
sampler = TPESampler(seed=SCRIPT_PARAMS['seed'])
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(xgb_evaluate, n_trials=SCRIPT_PARAMS['trials'], callbacks=[optuna_neptune_callback])
opt_utils.log_study_info(study)

study_importance = optuna.importance.get_param_importances(study)
study_importance = dt.Frame(variable=list(study_importance.keys()),
                            valor=list(study_importance.values()))
study_importance.to_csv(f'{ubicacion_datos_experimento}study_importance.csv')
neptune.log_artifact(f'{ubicacion_datos_experimento}study_importance.csv')

params = study.best_params
del params['prob_corte']
nrounds = all_nrounds[study.best_value]
neptune.set_property('best_nrounds', nrounds)

dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(params,
                  dtrain,
                  num_boost_round=nrounds,
                  feval=eval_ganancia,
                  maximize=True,
                  callbacks=[neptune_callback(max_num_features=15)])


def obtener_importance(tipo: str) -> pd.DataFrame:
    score = model.get_score(importance_type=tipo)
    return pd.DataFrame({'variable': list(score.keys()), tipo: list(score.values())})
    
    
tipos_scores = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
feature_importance = obtener_importance(tipos_scores.pop())
for tipo_score in tipos_scores:
    importance_score = obtener_importance(tipo_score)
    feature_importance = feature_importance.merge(importance_score, on='variable')
feature_importance = feature_importance.sort_values(by=['gain'], ascending=False)
feature_importance.to_csv(f'{ubicacion_datos_experimento}importance.csv')
neptune.log_artifact(f'{ubicacion_datos_experimento}importance.csv')

dataset_apply = dt.fread(SCRIPT_PARAMS['archivo_test'])
numeros_de_cliente = dataset_apply['numero_de_cliente']
dataset_apply = dataset_apply[:, dt.f[:].remove([dt.f.numero_de_cliente])]

dapply = xgb.DMatrix(dataset_apply)
y_pred = model.predict(dapply)

entrega = dt.Frame(numero_de_cliente=numeros_de_cliente,
                   estimulo=y_pred > study.best_params['prob_corte'])
entrega.to_csv(f'kaggle/{experimento}_xgboost.csv')
neptune.log_artifact(f'kaggle/{experimento}_xgboost.csv')
