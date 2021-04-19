import numpy as np
import lightgbm as lgb
import datatable as dt
import abc


class StudyModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, data: dt.Frame) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_feature_importance(self, variables: [str]) -> dt.Frame:
        pass

    @abc.abstractmethod
    def get_score(self) -> float:
        pass


class LightGbmModel(StudyModel):
    def __init__(self, model: lgb.Booster, name: str) -> None:
        self.model = model
        self.name = name

    def predict(self, data: dt.Frame) -> np.ndarray:
        return self.model.predict(data)

    def get_feature_importance(self, variables: [str]) -> dt.Frame:
        gain_importance = self.model.feature_importance(importance_type='gain')
        split_importance = self.model.feature_importance(importance_type='split')

        importance = dt.Frame(variable=variables,
                              gain=gain_importance,
                              split=split_importance)
        importance = importance.sort(-dt.f.gain)

        return importance

    def get_score(self) -> float:
        return self.model.best_score['validation']['ganancia']

