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

TRAIN_PARAMS = {
    'seed': 42,
    'trials': 10,
    'file_data': '../datasets/datos_fe_hist.gz',
    'max_foto_mes_train': 202002,
    'foto_mes_val': 202003,
    'max_foto_mes_entero': 202003,
    'foto_mes_kaggle': 202005,
}

np.random.seed(TRAIN_PARAMS['seed'])

if __name__ == '__main__':
    args = parser.parse_args()

    TRAIN_PARAMS['binaria_especial'] = args.binaria_especial
    TRAIN_PARAMS['model'] = args.model

    dataset = fread(TRAIN_PARAMS['file_data'])
    dataset = dataset[f.foto_mes <= TRAIN_PARAMS['max_foto_mes_entero'], :]
    dataset['azar'] = np.random.uniform(size=dataset.shape[0])
    dataset['clase01'] = dataset[:, ifelse(f.clase_ternaria == 'CONTINUA', 0, 1)]

    if args.binaria_especial:
        dataset['target'] = dataset[:, f.clase_ternaria != 'CONTINUA']
        dataset['weight'] = dataset[:, ifelse(f.clase_ternaria == 'BAJA+2', 1.0000001, 1)]
        campos_buenos = f[:].remove([f.clase_ternaria, f.target, f.azar, f.clase01, f.weight])
    else:
        dataset['target'] = dataset[:, f.clase_ternaria == 'BAJA+2']
        campos_buenos = f[:].remove([f.clase_ternaria, f.target, f.azar, f.clase01])

    X = dataset[(f.foto_mes <= TRAIN_PARAMS['max_foto_mes_train']) & ((f.clase01 == 1) | (f.azar < 0.1)), campos_buenos]
    y = dataset[(f.foto_mes <= TRAIN_PARAMS['max_foto_mes_train']) & ((f.clase01 == 1) | (f.azar < 0.1)), f.target]
    weights = None
    if args.binaria_especial:
        weights = dataset[(f.foto_mes <= TRAIN_PARAMS['max_foto_mes_train']) & ((f.clase01 == 1) | (f.azar < 0.1)), f.weight]

    X_val = dataset[f.foto_mes == TRAIN_PARAMS['foto_mes_val'], campos_buenos]
    y_val = dataset[f.foto_mes == TRAIN_PARAMS['foto_mes_val'], f.target]
    weights_val = None
    if args.binaria_especial:
        weights_val = dataset[f.foto_mes == TRAIN_PARAMS['foto_mes_val'], f.weight]

    dataset = None

    with Study(TRAIN_PARAMS) as study:
        if args.model == 'xgboost':
            optimizer = XGBoostOptimizer(X, y, weights, X_val, y_val, weights_val)
        else:
            optimizer = LightGBMOptimizer(X, y, weights, X_val, y_val, weights_val)

        study_importance = study.optimize(optimizer)
        study.log_csv(study_importance, f'{study.experiment_files_prefix}_study_importance.csv')

        best_models = optimizer.get_best_models()

        dataset = fread(TRAIN_PARAMS['file_data'])[:, f[:].remove([f.clase_ternaria])]
        dapply_stacking = dataset
        dapply_kaggle = dataset[f.foto_mes == TRAIN_PARAMS['foto_mes_kaggle'], :]

        for model in best_models:
            importance = model.get_feature_importance()
            study.log_csv(importance, f'{study.experiment_files_prefix}_{model}_importance.csv')

            y_pred = model.predict(dapply_stacking)
            apply_stacking = Frame(numero_de_cliente=dapply_stacking['numero_de_cliente'],
                                   foto_mes=dapply_stacking['foto_mes'],
                                   prob=y_pred,
                                   estimulo=y_pred > study.best_params['prob_corte'])
            study.log_csv(apply_stacking, f'{study.experiment_files_prefix}_{model}_stacking_apply.csv')

            y_pred = model.predict(dapply_kaggle)
            apply_kaggle = Frame(numero_de_cliente=dapply_kaggle['numero_de_cliente'],
                                 estimulo=y_pred > study.best_params['prob_corte'])
            study.log_csv(apply_kaggle, f'{study.experiment_files_prefix}_{model}_kaggle.csv')
