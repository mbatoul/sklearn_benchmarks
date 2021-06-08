# import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier

df_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv")
df_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")

df_train["dep_delayed_15min"] = np.where(df_train["dep_delayed_15min"] == "Y", 1, 0)
df_test["dep_delayed_15min"] = np.where(df_test["dep_delayed_15min"] == "Y", 1, 0)

# scikit-learn
X_train = df_train.drop(columns=["dep_delayed_15min"], axis=1).to_numpy()
y_train = df_train[["dep_delayed_15min"]].to_numpy()
X_test = df_test.drop(columns=["dep_delayed_15min"], axis=1).to_numpy()
y_test = df_test[["dep_delayed_15min"]].to_numpy()

model = HistGradientBoostingClassifier()
start = time.perf_counter()
model.fit(X_train, y_train)
end = time.perf_counter()
time_elapsed = round(end - start, 5)
print(f"time scikit-learn: {end - start}s")

y_pred = model.predict(X_test)
print(metrics.roc_auc_score(y_test, y_pred))

# TensorFlow Decision Forests
dtf_train = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="dep_delayed_15min")
dtf_test = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, label="dep_delayed_15min")

model = tfdf.keras.GradientBoostedTreesModel()
start = time.perf_counter()
model.fit(x=dtf_train)
end = time.perf_counter()
time_elapsed = round(end - start, 5)
print(f"time scikit-learn: {end - start}s")

y_pred = model.predict(dtf_test)
y_test = df_test["dep_delayed_15min"]
print(metrics.roc_auc_score(y_test, y_pred))
