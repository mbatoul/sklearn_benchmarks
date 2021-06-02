# Load TensorFlow Decision Forests
import tensorflow_decision_forests as tfdf
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(ds_train)

# Evaluate the model
model.compile(metrics=["accuracy"])
print(model.evaluate(ds_test))
