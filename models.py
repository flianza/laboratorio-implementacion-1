from typing import Union
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import datatable as dt
import pandas as pd
import abc


class StudyModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, data: dt.Frame) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_feature_importance(self) -> Union[dt.Frame, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_score(self) -> float:
        pass

    def __str__(self):
        return f'{self.__class__.__name__.replace("Model", "")}_{int(self.get_score())}'


class LightGBMModel(StudyModel):
    def __init__(self, model: lgb.Booster) -> None:
        self.model = model

    def predict(self, data: dt.Frame) -> np.ndarray:
        return self.model.predict(data)

    def get_feature_importance(self) -> dt.Frame:
        gain_importance = self.model.feature_importance(importance_type='gain')
        split_importance = self.model.feature_importance(importance_type='split')

        importance = dt.Frame(variable=self.model.feature_name(),
                              gain=gain_importance,
                              split=split_importance)
        importance = importance.sort(-dt.f.gain)

        return importance

    def get_score(self) -> float:
        return self.model.best_score['validation']['ganancia']


class XGBoostModel(StudyModel):
    def __init__(self, model: xgb.Booster) -> None:
        self.model = model

    def predict(self, data: dt.Frame) -> np.ndarray:
        return self.model.predict(xgb.DMatrix(data))

    def get_feature_importance(self) -> dt.Frame:
        def __obtener_importance(tipo: str) -> pd.DataFrame:
            score = self.model.get_score(importance_type=tipo)
            return pd.DataFrame({'variable': list(score.keys()), tipo: list(score.values())})

        tipos_scores = ['weight', 'gain']
        importance = __obtener_importance(tipos_scores.pop())
        for tipo_score in tipos_scores:
            importance_score = __obtener_importance(tipo_score)
            importance = importance.merge(importance_score, on='variable')
        importance = importance.sort_values(by=['gain'], ascending=False)
        importance = importance.set_index('variable')

        return importance

    def get_score(self) -> float:
        return self.model.best_score
