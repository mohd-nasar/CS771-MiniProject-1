import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# Load the training and validation datasets
train_df = pd.read_csv("train_text_seq.csv")
valid_df = pd.read_csv("valid_text_seq.csv")

# Preprocess the data
def preprocess_data(df):
    df['processed_str'] = df['input_str'].apply(lambda x: x[3:])  # Remove leading zeros
    # No substring removal
    df['modified_str'] = df['processed_str']
    return df

# Preprocess train and validation data
train_df = preprocess_data(train_df)
valid_df = preprocess_data(valid_df)

# Tokenizer for text data (digit sequences)
def tokenize_sequences(train_df, valid_df):
    tokenizer = Tokenizer(char_level=True)  # Char-level tokenizer for digit sequences
    tokenizer.fit_on_texts(train_df['modified_str'])
    X_train_seq = tokenizer.texts_to_sequences(train_df['modified_str'])
    X_valid_seq = tokenizer.texts_to_sequences(valid_df['modified_str'])

    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, padding='post')
    X_valid_padded = pad_sequences(X_valid_seq, padding='post', maxlen=X_train_padded.shape[1])

    return X_train_padded, X_valid_padded, tokenizer

# Tokenize the training and validation sequences
X_train_padded, X_valid_padded, tokenizer = tokenize_sequences(train_df, valid_df)

# Encode labels for binary classification
y_train = train_df['label'].values

# Define the RNN model
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size from tokenizer
max_sequence_length = X_train_padded.shape[1]  # Sequence length

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_sequence_length))  # Reduced embedding dimension
model.add(Bidirectional(GRU(32, return_sequences=True)))  # Bidirectional GRU
model.add(Dropout(0.2))  # Dropout for regularization
model.add(GRU(16))  # Reduced number of GRU units
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(8, activation='relu'))  # Smaller fully connected layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the RNN model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Different fractions of the validation dataset to evaluate
validation_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
validation_accuracies = []

# Loop through different fractions of the validation dataset
for frac in validation_fractions:
    # Select the subset of the validation data based on the current fraction
    subset_size = int(frac * valid_df.shape[0])
    if subset_size == 0:  # Ensure subset size is at least 1
        subset_size = 1
    X_valid_subset = X_valid_padded[:subset_size]
    y_valid_subset = valid_df['label'].values[:subset_size]

    # Train the RNN model using the full training dataset
    history = model.fit(X_train_padded, y_train, epochs=10, batch_size=32,
                        validation_data=(X_valid_subset, y_valid_subset),
                        callbacks=[early_stopping], verbose=0)

    # Evaluate the model on the current validation subset
    val_loss, val_accuracy = model.evaluate(X_valid_subset, y_valid_subset, verbose=0)
    validation_accuracies.append(val_accuracy)

    print(f"Validation Accuracy with {int(frac * 100)}% of validation data: {val_accuracy:.4f}")

# Plotting validation accuracy against percentage of validation data used
plt.figure(figsize=(10, 6))
plt.plot([int(frac * 100) for frac in validation_fractions], validation_accuracies, marker='o', color='blue')
plt.xlabel('Percentage of Validation Data Used (%)')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs Percentage of Validation Data')
plt.xticks(validation_fractions * 100)
plt.grid(True)
plt.show()

# Print the number of trainable parameters
trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
print(f"Number of trainable parameters: {trainable_params}")

# Final Model Accuracy with Full Validation Data
print(f"Final Model Validation Accuracy with Full Validation Data: {validation_accuracies[-1] * 100:.2f}%")
