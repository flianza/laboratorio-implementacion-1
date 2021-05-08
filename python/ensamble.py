import argparse
import numpy as np
import os
from datatable import fread, f, ifelse, Frame, rowsum

parser = argparse.ArgumentParser()
parser.add_argument('--version')
parser.add_argument("--experimentos", nargs="+", default=[])

PARAMS = {
    'foto_mes_val': 202003,
    'foto_mes_kaggle': 202005
}


def leer_dataset(experimentos: list, foto_mes: int) -> Frame:
    dataset = None
    for experimento in experimentos:
        for file in os.listdir(f'../experimentos/{experimento}/'):
            if file.endswith('stacking_apply.csv'):
                campos_buenos = ['numero_de_cliente', 'estimulo']
                stacking = fread(f'../experimentos/{experimento}/{file}')[f.foto_mes == foto_mes, :].sort(['foto_mes', 'numero_de_cliente'])[:, campos_buenos]
                if dataset is None:
                    dataset = stacking
                    dataset.names = {'estimulo': f'{experimento}_estimulo'}
                else:
                    dataset[f'{experimento}_estimulo'] = stacking[:, 'estimulo']
                break
    dataset['votos'] = dataset[:, rowsum(f[:].remove([f.numero_de_cliente]))]
    return dataset[:, ['numero_de_cliente', 'votos']]


def leer_clase_ternaria(foto_mes: int) -> Frame:
    archivo = '../datasetsOri/paquete_premium.txt.gz'
    return fread(archivo)[f.foto_mes == foto_mes, :].sort(['foto_mes', 'numero_de_cliente'])[:, 'clase_ternaria']


def calcular_ganancia(data: Frame, votos: int) -> float:
    datos_votos = data[:, 'votos'].to_numpy().flatten() >= votos
    ganancias = data[:, ifelse(f.clase_ternaria == 'BAJA+2', 29250, -750)].to_numpy().flatten()
    return np.dot(datos_votos, ganancias)


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_val = leer_dataset(args.experimentos, PARAMS['foto_mes_val'])
    dataset_val['clase_ternaria'] = leer_clase_ternaria(PARAMS['foto_mes_val'])
    dataset_apply = leer_dataset(args.experimentos, PARAMS['foto_mes_kaggle'])

    best_votes = 1
    best_ganancia = -np.inf
    for i in range(1, len(args.experimentos) + 1):
        ganancia = calcular_ganancia(dataset_val, i)
        if ganancia > best_ganancia:
            best_ganancia = ganancia
            best_votes = i

    dataset_apply['estimulo'] = dataset_apply[:, f.votos >= best_votes]
    dataset_apply = dataset_apply[:, [f.numero_de_cliente, f.estimulo]]
    dataset_apply.to_csv(f'../ensemble_{best_ganancia}.csv')

