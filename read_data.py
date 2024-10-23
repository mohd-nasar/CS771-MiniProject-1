import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load Data
train_emoticon_df = pd.read_csv("train_emoticon.csv")
valid_emoticon_df = pd.read_csv("test_emoticon.csv")

# Extract input emoticons and labels
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()  # Use 'label' instead of 'Label'
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_Y = valid_emoticon_df['label'].tolist()  # Use 'label' instead of 'Label'

# Tokenize emoticons
tokenizer = Tokenizer(char_level=True)  # char_level=True for character-wise tokenization
tokenizer.fit_on_texts(train_emoticon_X + valid_emoticon_X)

# Convert emoticons to sequences of token indices
train_sequences = tokenizer.texts_to_sequences(train_emoticon_X)
valid_sequences = tokenizer.texts_to_sequences(valid_emoticon_X)

# Pad sequences to ensure they are of equal length
max_length = max(len(seq) for seq in train_sequences + valid_sequences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding='post')

# Define RNN model
vocab_size = len(tokenizer.word_index) + 1  # Plus 1 for padding token

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length),
    LSTM(64, return_sequences=False),  # LSTM layer
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split the training data for internal validation
X_train, X_val, y_train, y_val = train_test_split(train_padded, train_emoticon_Y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(valid_padded, valid_emoticon_Y)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Plot train/validation accuracy over epochs
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction on validation set
y_valid_pred = (model.predict(valid_padded) > 0.5).astype("int32")
print(f"Validation Accuracy (sklearn): {accuracy_score(valid_emoticon_Y, y_valid_pred):.4f}")
