import argparse
import datatable as dt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from optimizers import XGBoostOptimizer, LightGBMOptimizer
from studies import Study

parser = argparse.ArgumentParser()

parser.add_argument('--model')
parser.add_argument('--binaria-especial', dest='binaria_especial', action='store_true')
parser.add_argument('--binaria-comun', dest='binaria_especial', action='store_false')
parser.set_defaults(binaria_especial=True)

if __name__ == '__main__':
    args = parser.parse_args()

    seed = 42
    trials = 2
    file_to_train = 'datasets/201905_fechas_drifting_montos.csv'
    file_to_predict = 'datasets/201907_fechas_drifting_montos.csv'
    np.random.seed(seed)

    dataset = dt.fread(file_to_train)

    if args.binaria_especial:
        dataset['target'] = dataset[:, dt.f.clase_ternaria != 'CONTINUA']
        weight_dataset = dataset[:, dt.ifelse(dt.f.clase_ternaria == 'BAJA+2', 1.0000001, 1)]
    else:
        dataset['target'] = dataset[:, dt.f.clase_ternaria == 'BAJA+2']
        weight_dataset = None

    X_dataset = dataset[:, dt.f[:].remove([dt.f.clase_ternaria, dt.f.target])]
    y_dataset = dataset[:, 'target']

    train, test = StratifiedKFold(n_splits=2).split(X_dataset, y_dataset)
    X = X_dataset[train, :]
    y = y_dataset[train, :]
    weights = weight_dataset[train, :]
    X_val = X_dataset[test, :]
    y_val = y_dataset[test, :]
    weights_val = weight_dataset[test, :]

    with Study(file_to_train, file_to_predict, trials, seed) as study:
        if args.model == 'xgboost':
            optimizer = XGBoostOptimizer(X, y, weights, X_val, y_val, weights_val)
        else:
            optimizer = LightGBMOptimizer(X, y, weights, X_val, y_val, weights_val)

        study_importance = study.optimize(optimizer)
        study.log_csv(study_importance, f'{study.experiment_files_prefix}_study_importance.csv')

        best_models = optimizer.get_best_models(2)

        for model in best_models:
            importance = model.get_feature_importance()
            study.log_csv(importance, f'{study.experiment_files_prefix}_{model}_importance.csv')

            dapply = dt.fread(file_to_predict)
            y_pred = model.predict(dapply)

            apply = dt.Frame(numero_de_cliente=dapply['numero_de_cliente'],
                             prob=y_pred,
                             estimulo=y_pred > study.best_params['prob_corte'])
            study.log_csv(apply, f'{study.experiment_files_prefix}_{model}_apply.csv')
