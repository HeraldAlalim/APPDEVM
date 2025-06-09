# sign_language_motion.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load training data
X_train = np.load('X_train.npy')  # shape (num_samples, seq_length, num_features)
y_train = np.load('y_train.npy')  # shape (num_samples,)

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

num_classes = len(np.unique(y_train))

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Define LSTM model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model((X_train.shape[1], X_train.shape[2]), num_classes)

# Save best model during training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train model
model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Load best model weights
model.load_weights('best_model.h5')

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test_cat)
print(f'Test accuracy: {acc*100:.2f}%')

# Predict function example for new sequences
def predict_sequence(model, sequence):
    """
    sequence: numpy array of shape (seq_length, num_features)
    """
    sequence = np.expand_dims(sequence, axis=0)  # shape (1, seq_length, num_features)
    probs = model.predict(sequence)[0]
    pred_class = np.argmax(probs)
    return pred_class, probs

if __name__ == '__main__':
    # Example: Predict the first test sample
    pred, probs = predict_sequence(model, X_test[0])
    print(f'Predicted class: {pred}, Probabilities: {probs}')
