import tensorflow_decision_forests as tfdf
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf

X, y = make_classification(
    n_samples=100_000,
    n_features=1000,
    n_classes=5,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = pd.DataFrame(X_train)
X_train["label"] = y_train

X_test = pd.DataFrame(X_test)
X_test["label"] = y_test

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label="label")

model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label="label")

model.compile(metrics=["accuracy"])
print(model.evaluate(test_ds))
