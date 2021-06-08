import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv")
df_test = pd.read_csv("https://s3.amazonaws.com/benchm-ml--main/test.csv")

df_train["dep_delayed_15min"] = np.where(df_train["dep_delayed_15min"] == "Y", 1, 0)
df_test["dep_delayed_15min"] = np.where(df_test["dep_delayed_15min"] == "Y", 1, 0)

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
df_train[cat_cols] = df_train[cat_cols].apply(LabelEncoder().fit_transform)
df_test[cat_cols] = df_test[cat_cols].apply(LabelEncoder().fit_transform)

# scikit-learn
X_train = df_train.drop(columns=["dep_delayed_15min"], axis=1).to_numpy()
y_train = df_train["dep_delayed_15min"].values
X_test = df_test.drop(columns=["dep_delayed_15min"], axis=1).to_numpy()
y_test = df_test["dep_delayed_15min"].values

times = []
scores = []
for _ in range(10):
    model = HistGradientBoostingClassifier()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    end = time.perf_counter()

    time_elapsed = round(end - start, 3)
    times.append(time_elapsed)

    y_pred = model.predict(X_test)
    score = round(metrics.roc_auc_score(y_test, y_pred), 3)
    scores.append(score)

print(f"average time scikit-learn: {np.mean(times)}s")
print(f"average score scikit-learn: {np.mean(scores)}")

# TensorFlow Decision Forests
dtf_train = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="dep_delayed_15min")
dtf_test = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, label="dep_delayed_15min")

times = []
scores = []
for _ in range(10):
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=100,
        shrinkage=0.1,
        early_stopping="NONE",
        # validation_ratio=0.05,
        # early_stopping_num_trees_look_ahead=10,
    )
    start = time.perf_counter()
    model.fit(x=dtf_train)
    end = time.perf_counter()

    time_elapsed = round(end - start, 3)
    times.append(time_elapsed)

    y_pred = model.predict(dtf_test)
    score = round(metrics.roc_auc_score(y_test, y_pred), 3)
    scores.append(score)

print(f"average time tfdf: {np.mean(times)}s")
print(f"average score tfdf: {np.mean(scores)}")
