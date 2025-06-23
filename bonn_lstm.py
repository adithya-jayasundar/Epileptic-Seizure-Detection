import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt

# Constants
DATASET_PATH = r'C:\Users\HP\Desktop\DUK\sem_2\computation\eeg_project\Bonn Univeristy Dataset'
FOLDERS = ['F', 'N', 'O', 'S', 'Z']
LABELS = {'F': 0, 'N': 0, 'O': 0, 'Z': 0, 'S': 1}
ORIGINAL_LENGTH = 4097
NEW_LENGTH = 4096
SEGMENT_SIZE = 256
SEGMENTS_PER_FILE = NEW_LENGTH // SEGMENT_SIZE

# Storage
raw_segments = []
labels = []

# Preprocessing
for folder in FOLDERS:
    folder_path = os.path.join(DATASET_PATH, folder)
    label = LABELS[folder]

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.TXT'):
            file_path = os.path.join(folder_path, filename)
            signal = np.loadtxt(file_path)

            if len(signal) == ORIGINAL_LENGTH:
                signal = signal[:-1]

            segments = np.split(signal, SEGMENTS_PER_FILE)

            for seg in segments:
                scaler = StandardScaler()
                seg_normalized = scaler.fit_transform(seg.reshape(-1, 1)).flatten()
                raw_segments.append(seg_normalized)
                labels.append(label)

print("Preprocessing done!")

# Convert to arrays
X_raw = np.array(raw_segments)  # (samples, 256)
y = np.array(labels)            # (samples,)

# Train/Val/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Model Definition
def build_lstm_model():
    input_raw = Input(shape=(256,))
    x = layers.Reshape((256, 1))(input_raw)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='sigmoid')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='sigmoid')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_raw, outputs=x)
    return model

model = build_lstm_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.show()

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save model
model.save('lstm_only_eeg_model.h5')

# Load (if needed)
model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\DUK\sem_2\computation\eeg_project\lstm_only_eeg_model.h5')

# Prediction Function
def predict_eeg_lstm(filepath):
    signal = np.loadtxt(filepath)

    if len(signal) == ORIGINAL_LENGTH:
        signal = signal[:-1]

    segments = np.split(signal, SEGMENTS_PER_FILE)
    all_preds = []

    for seg in segments:
        scaler = StandardScaler()
        seg_norm = scaler.fit_transform(seg.reshape(-1, 1)).flatten()
        seg_input = np.expand_dims(seg_norm, axis=0)  # (1, 256)
        pred = model.predict(seg_input, verbose=0)
        all_preds.append(pred[0][0])

    final_score = np.mean(all_preds)
    print(f"Average prediction score: {final_score:.4f}")

    if final_score > 0.5:
        print("Prediction: 🚨 Epileptic Seizure Detected")
    else:
        print("Prediction: ✅ Non-Seizure (Normal Activity)")

    return final_score

# Example usage
predict_eeg_lstm(r'C:\Users\HP\Desktop\DUK\sem_2\computation\eeg_project\Bonn Univeristy Dataset\O\O017.txt')  # Change path
