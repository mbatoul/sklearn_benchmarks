import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
from sklearn.base import BaseEstimator, ClassifierMixin


class TFDFGradBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model_ = tfdf.keras.GradientBoostedTreesModel(**kwargs)

    def fit(self, X, y):
        df_train = pd.DataFrame(data=X)
        df_train["label"] = y
        dtf_train = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="label")
        self.model_.fit(x=dtf_train)
        return self

    def predict(self, X):
        df_test = pd.DataFrame(data=X)
        dtf_test = tfdf.keras.pd_dataframe_to_tf_dataset(df_test)
        return self.model_.predict(dtf_test)
