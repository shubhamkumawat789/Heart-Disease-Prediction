import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

model_loaded = keras.models.load_model("artifacts/heart_mlp_tf.keras")
preprocessor_loaded = joblib.load("artifacts/preprocessor.joblib")

df = pd.read_csv("Heart_Disease.csv")

df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num", "id"])

x = df.drop(columns=["target"])
y = df["target"]
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

sample = x_test_raw.iloc[[0]]

x = preprocessor_loaded.transform(sample)
if hasattr(x, "toarray"):
    x = x.toarray()
x = x.astype("float32")

prob = model_loaded.predict(x)[0][0]
pred = int(prob >= 0.5)

print("Prediction:", pred)
print("Probability:", prob)
print("Actual:", int(y_test.iloc[0]))
