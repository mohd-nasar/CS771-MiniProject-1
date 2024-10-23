Ensure to install the following libraries
pip install numpy
pip install pandas
pip install scikit-learn
pip install scipy
pip install pytorch
pip install tensorflow
pip install keras
pip install matplotlib
pip install seaborn



Overview
This project focuses on Emoticon Classification using machine learning techniques such as Logistic Regression, LSTM, and RandomForest. The goal is to accurately classify emoticons based on sequential data using various models and feature extraction methods.

Data Description
The dataset consists of three main files:

train_emoticon.csv
valid_emoticon.csv
test_emoticon.csv
Each of these files contains sequences of emoticons along with corresponding binary labels.

Data Preprocessing
We performed the following preprocessing steps:

Emoji Removal: Removed common emojis to clean the data.
Tokenization: Converted emoticons to sequences of integer indices.
Padding: Padded sequences to maintain uniform length.
Label Encoding: Emoticon labels were encoded as integers.
Models Implemented
1. RNN for Feature Extraction and Logistic Regression
An RNN-based model was used for feature extraction with the following architecture:

Embedding Layer: Maps each character to a 32-dimensional dense vector.
LSTM Layer: A single LSTM layer with 2 units.
Output Layer: A Dense unit with sigmoid activation for binary classification.
The model was trained using the Adam optimizer and binary cross-entropy loss for 50 epochs with a batch size of 32. Validation accuracy was measured across various training percentages.

Validation Accuracy (100% Training Data): 96.84%
Trainable Parameters: 7,824
2. RandomForest for Feature Prediction
We also employed a RandomForest classifier using data in NPZ format (13x786 embeddings):

n_estimators: 10

max_depth: 10

min_samples_split: 10

min_samples_leaf: 5

Validation Accuracy (100% Training Data): 97%

Parameters (Nodes): 3,042

3. LSTM for Sequence Classification
The LSTM-based model is designed to capture sequential dependencies within text sequences:

Embedding Layer: 32-dimensional dense vector for each character.

LSTM Layer: Single LSTM layer with 32 units.

Dense Layers: Two Dense layers with 16 and 8 units using ReLU activation.

Validation Accuracy (100% Training Data): 86.92%

Trainable Parameters: 7,922

4. Logistic Regression with PCA
Logistic regression was implemented with L1 regularization and Principal Component Analysis (PCA) for dimensionality reduction. The regularization parameter C was tuned to find the optimal value.

Validation Accuracy (with PCA, C = 1): 92%
Graphical Insights
Training and validation accuracies for different models and training data percentages were plotted.
Confusion matrices were generated to analyze classification performance.
Conclusion
The combination of PCA and Logistic Regression provided robust performance, achieving 92% validation accuracy. The RNN and RandomForest models also demonstrated strong results, making this a successful experiment in emoticon classification.

Authors
Mohd Nasar Siddiqui (220661)
Abhishek Kumar (220044)
Param Soni (220752)
Kartik (220503)
