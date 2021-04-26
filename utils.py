import os
import functools
import time
import neptune
import numpy as np
from pathlib import Path
from typing import Tuple


def get_score_corte(prob=0.025) -> float:
    return np.log(prob / (1 - prob))


def start_experiment(params: dict) -> Tuple[str, str]:
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZTBhNWNhZS0xNTNkLTQ2NjktODI0ZC1kOTAyMzhmNzllNDAifQ=='
    neptune.init('flianza/laboratorio-1-mes', api_token=api_token)
    neptune.create_experiment('lightgbm-tuning', tags=['lightgbm', 'tuning', 'optuna'], params=params)

    experiment_number = neptune.get_experiment().id

    experiments_folder = 'experimentos'
    experiment_path = f'{experiments_folder}/{experiment_number}'
    if not Path(experiments_folder).exists():
        os.mkdir(experiments_folder)
    os.mkdir(experiment_path)
    return experiment_number, f'{experiment_path}/{experiment_number}'


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f'Completado: {func.__name__!r} en {duration:.4f}segs')
        return value
    return wrapper_timer
