# Load TensorFlow Decision Forests
import tensorflow_decision_forests as tfdf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1e5,
    n_features=1e3,
    n_classes=5,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

X_train, y_train, X_test, y_test = train_test_split(X, y)

X_train["label"] = y_train
X_test["label"] = y_test

# Convert the pandas dataframe into a TensorFlow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label="species")

# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Convert it to a TensorFlow dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label="species")

# Evaluate the model
model.compile(metrics=["accuracy"])
print(model.evaluate(test_ds))
