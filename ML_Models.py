import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the datasets
train_data = np.load('train_feature.npz')
valid_data = np.load('valid_feature.npz')
test_data = np.load('test_feature.npz')

# Extract features and labels from the datasets
X_train_full = train_data['features']
y_train_full = train_data['label']
X_valid = valid_data['features']
y_valid = valid_data['label']
X_test = test_data['features']

# Ensure features are 2D arrays by flattening them if necessary
X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Print shapes to verify correctness
print(f"Training features shape: {X_train_full.shape}")
print(f"Training labels shape: {y_train_full.shape}")
print(f"Validation features shape: {X_valid.shape}")
print(f"Validation labels shape: {y_valid.shape}")
print(f"Test features shape: {X_test.shape}")

# Define the percentages of training data to be used
percentages = [0.2, 0.4, 0.6, 0.8, 1.0]

# Initialize lists to store accuracies and number of parameters
rf_train_accuracies = []
rf_valid_accuracies = []
rf_num_parameters_list = []

# Function to count the number of parameters in a RandomForest model
def count_rf_parameters(model):
    return sum(tree.tree_.node_count for tree in model.estimators_)

# Define RandomForest hyperparameters
rf_params = {
    'n_estimators': 10,      # Number of trees in the forest
    'max_depth': 10,         # Maximum depth of each tree
    'min_samples_split': 10, # Minimum samples to split an internal node
    'min_samples_leaf': 5    # Minimum samples required to be a leaf node
}

# Train the RandomForestClassifier with varying data sizes
for pct in percentages:
    size = int(pct * len(X_train_full))
    X_train_subset = X_train_full[:size]
    y_train_subset = y_train_full[:size]

    # Train the RandomForest model
    rf_model = RandomForestClassifier(**rf_params, random_state=42)
    rf_model.fit(X_train_subset, y_train_subset)

    # Calculate training accuracy
    y_rf_train_pred = rf_model.predict(X_train_subset)
    train_accuracy_rf = accuracy_score(y_train_subset, y_rf_train_pred)
    rf_train_accuracies.append(train_accuracy_rf)

    # Calculate validation accuracy
    y_rf_valid_pred = rf_model.predict(X_valid)
    valid_accuracy_rf = accuracy_score(y_valid, y_rf_valid_pred)
    rf_valid_accuracies.append(valid_accuracy_rf)

    # Count the number of parameters in the RandomForest model
    rf_num_parameters = count_rf_parameters(rf_model)
    rf_num_parameters_list.append(rf_num_parameters)

    # Generate predictions on the test data
    test_predictions = rf_model.predict(X_test)

    # Save the predictions to a text file without any labels
    test_predictions_df = pd.DataFrame(test_predictions)  # Create a DataFrame
    test_predictions_df.to_csv('pred_deepfeat.txt', header=False, index=False)  # Save to .txt without header

    print("Test predictions saved to 'pred_deepfeat.txt'.")

# Plot: Training vs Validation Accuracy for RandomForest
plt.subplot(2, 2, 2)
plt.plot([int(p * 100) for p in percentages], rf_train_accuracies, marker='o', label='Training Accuracy')
plt.plot([int(p * 100) for p in percentages], rf_valid_accuracies, marker='s', label='Validation Accuracy')
plt.title('RandomForest: Training vs Validation Accuracy')
plt.xlabel('Percentage of Training Data Used (%)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Annotate the number of parameters on the plot
for i, txt in enumerate(rf_num_parameters_list):
    plt.annotate(f'Params: {txt}', (int(percentages[i] * 100), rf_valid_accuracies[i]))

plt.tight_layout()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_valid, y_rf_valid_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_valid), yticklabels=np.unique(y_valid))
plt.title('Confusion Matrix for RandomForest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
