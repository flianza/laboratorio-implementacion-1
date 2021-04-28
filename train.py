import argparse
import numpy as np
from datatable import fread, f, ifelse, Frame

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
    trials = 10
    file_data = 'datasets/datos_fe_hist.gz'
    np.random.seed(seed)

    dataset = fread(file_data)
    dataset = dataset[f.foto_mes <= 202003]

    if args.binaria_especial:
        dataset['target'] = dataset[:, f.clase_ternaria != 'CONTINUA']
        weight_dataset = dataset[:, ifelse(f.clase_ternaria == 'BAJA+2', 1.0000001, 1)]
    else:
        dataset['target'] = dataset[:, f.clase_ternaria == 'BAJA+2']
        weight_dataset = None

    X_dataset = dataset[:, f[:].remove([f.clase_ternaria, f.target])]
    y_dataset = dataset[:, 'target']

    X = X_dataset[f.foto_mes <= 202002, :]
    y = y_dataset[f.foto_mes <= 202002, :]
    weights = weight_dataset[f.foto_mes <= 202002, :]
    X_val = X_dataset[f.foto_mes == 202003, :]
    y_val = y_dataset[f.foto_mes == 202003, :]
    weights_val = weight_dataset[f.foto_mes == 202003, :]

    with Study(202002, 202003, trials, seed) as study:
        if args.model == 'xgboost':
            optimizer = XGBoostOptimizer(X, y, weights, X_val, y_val, weights_val)
        else:
            optimizer = LightGBMOptimizer(X, y, weights, X_val, y_val, weights_val)

        study_importance = study.optimize(optimizer)
        study.log_csv(study_importance, f'{study.experiment_files_prefix}_study_importance.csv')

        best_models = optimizer.get_best_models(5)

        for model in best_models:
            importance = model.get_feature_importance()
            study.log_csv(importance, f'{study.experiment_files_prefix}_{model}_importance.csv')

            dapply = fread(file_data)[f.foto_mes == 202005, :]
            y_pred = model.predict(dapply)

            apply = Frame(numero_de_cliente=dapply['numero_de_cliente'],
                          prob=y_pred,
                          estimulo=y_pred > study.best_params['prob_corte'])
            study.log_csv(apply, f'{study.experiment_files_prefix}_{model}_apply.csv')
