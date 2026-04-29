# ==============================
# Breast Cancer Classification
# Using Neural Network
# ==============================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ==============================
# Load Dataset
# ==============================

breast_cancer = sklearn.datasets.load_breast_cancer()

# Convert to DataFrame
data = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)

# Add Target Column
data['label'] = breast_cancer.target

print(data.head())

# ==============================
# Features and Target
# ==============================
X = data.drop(columns='label')
Y = data['label']
# ==============================
# Train Test Split
# ==============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

# ==============================
# Standardization
# ==============================

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ==============================
# Build Neural Network
# ==============================

model = keras.Sequential([

    keras.layers.Dense(20, activation='relu', input_shape=(30,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')

])

# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# Train Model
# ==============================

history = model.fit(
    X_train_std,
    Y_train,
    validation_split=0.1,
    epochs=20
)

# ==============================
# Evaluate Model
# ==============================

loss, accuracy = model.evaluate(X_test_std, Y_test)

print("\nTest Accuracy:", accuracy)

# ==============================
# Plot Accuracy
# ==============================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# ==============================
# Plot Loss
# ==============================

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# ==============================
# Prediction System
# ==============================

input_data = (
    11.76,21.6,74.72,427.9,0.08637,
    0.04966,0.01657,0.01115,0.1495,0.05888,
    0.4062,1.21,2.635,28.47,0.005857,
    0.009758,0.01168,0.007445,0.02406,0.001769,
    12.98,25.72,82.98,516.5,0.1085,
    0.08615,0.05523,0.03715,0.2433,0.06563
)

# Convert to NumPy Array
input_array = np.asarray(input_data)

# Reshape
input_reshaped = input_array.reshape(1, -1)

# Standardize
input_std = scaler.transform(input_reshaped)

# Predict
prediction = model.predict(input_std)

predicted_label = np.argmax(prediction)

print("\nPrediction Result:")

if predicted_label == 0:
    print("Malignant Tumor")
else:
    print("Benign Tumor")