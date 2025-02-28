import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from gensim.models import Word2Vec

# i am using 2 datasets - one curated for gendered speech and one for sentiment analysis
df_bias = pd.read_csv("hf://datasets/flax-sentence-embeddings/Gender_Bias_Evaluation_Set/bias_evaluation.csv")
splits = {'train': 'train_df.csv', 'validation': 'val_df.csv', 'test': 'test_df.csv'}
df_sentiment = pd.read_csv("hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/" + splits["train"])

# i am training word2vec model on all text so vocabulary has everything
combined_text = df_bias['base_sentence'].tolist() + df_bias['male_sentence'].tolist() + df_bias['female_sentence'].tolist() + df_sentiment['text'].tolist()
word2vec_model = Word2Vec(combined_text, vector_size=100, window=3, min_count=1, workers=4, epochs=5)

# creating embedding matrix
embedding_dim = 100
word_index = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    embedding_matrix[i] = word2vec_model.wv[word]

# converts sentences to sequences of indices
def sentences_to_indices(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = [word_index[word] for word in sentence if word in word_index]
        sequences.append(sequence)
    return sequences

# data for gender prediction model
X_gender = sentences_to_indices(df_bias['base_sentence'].tolist() + df_bias['male_sentence'].tolist() + df_bias['female_sentence'].tolist(), word_index)
X_gender = tf.keras.preprocessing.sequence.pad_sequences(X_gender, maxlen=100)
y_gender = np.array([0.5] * len(df_bias) + [0] * len(df_bias) + [1] * len(df_bias))
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)

# gender prediction CNN
input_layer_gender = Input(shape=(100,))
embedding_layer_gender = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_layer_gender)
conv_layer_gender = Conv1D(128, 5, activation='relu')(embedding_layer_gender)
pooling_layer_gender = GlobalMaxPooling1D()(conv_layer_gender)
output_layer_gender = Dense(1, activation='sigmoid')(pooling_layer_gender)

gender_model = Model(inputs=input_layer_gender, outputs=output_layer_gender)
gender_model.compile(loss='binary_crossentropy', metrics=['accuracy'])
gender_model.fit(X_train_gender, y_train_gender, epochs=10, batch_size=32, validation_split=0.2)

# evaluate gender predictions
y_pred_gender = gender_model.predict(X_test_gender)
y_pred_gender = (y_pred_gender > 0.5).astype(int)
y_test_gender = (y_test_gender > 0.51).astype(int)
accuracy_gender = accuracy_score(y_test_gender, y_pred_gender)
f1_gender = f1_score(y_test_gender, y_pred_gender)
print(f"Gender Prediction - Accuracy: {accuracy_gender}, F1-Score: {f1_gender}")

# data for sentiment analysis model
X_sentiment = sentences_to_indices(df_sentiment['text'].tolist(), word_index)
X_sentiment = tf.keras.preprocessing.sequence.pad_sequences(X_sentiment, maxlen=100)
y_sentiment = df_sentiment['label'].values
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X_sentiment, y_sentiment, test_size=0.2, random_state=42)

# sentiment analysis CNN
input_layer_sentiment = Input(shape=(100,))
embedding_layer_sentiment = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)(input_layer_sentiment)
conv_layer_sentiment = Conv1D(128, 5, activation='relu')(embedding_layer_sentiment)
pooling_layer_sentiment = GlobalMaxPooling1D()(conv_layer_sentiment)
output_layer_sentiment = Dense(3, activation='softmax')(pooling_layer_sentiment)  # 3 classes for sentiment

sentiment_model = Model(inputs=input_layer_sentiment, outputs=output_layer_sentiment)
sentiment_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# predicting gender for sentiment analysis training data
y_train_gender_sentiment = gender_model.predict(X_train_sentiment)
y_train_gender_sentiment = (y_train_gender_sentiment > 0.5).astype(int)  # 0 for male, 1 for female

# get sample weights based on predicted gender labels
sample_weights = compute_sample_weight('balanced', y_train_gender_sentiment)

# train sentiment analysis model with sample weights
sentiment_model.fit(X_train_sentiment, y_train_sentiment, epochs=10, batch_size=32, validation_split=0.2, sample_weight=sample_weights)

# evaluate sentiment analysis model
y_pred_sentiment = sentiment_model.predict(X_test_sentiment)
y_pred_sentiment = np.argmax(y_pred_sentiment, axis=1)  # convert probabilities to class labels
accuracy_sentiment = accuracy_score(y_test_sentiment, y_pred_sentiment)
f1_sentiment = f1_score(y_test_sentiment, y_pred_sentiment, average='weighted')
print(f"Sentiment Analysis - Accuracy: {accuracy_sentiment}, F1-Score: {f1_sentiment}")

############################# DEBIASING #############################

# Predict gender for the test sentences
y_pred_gender = gender_model.predict(X_test_sentiment)
y_pred_gender = (y_pred_gender > 0.5).astype(int)  # 0 for male, 1 for female

# calculate sentiment bias
male_indices = np.where(y_pred_gender == 0)[0]
female_indices = np.where(y_pred_gender == 1)[0]
male_sentiment = y_pred_sentiment[male_indices]
female_sentiment = y_pred_sentiment[female_indices]
male_positive_ratio = np.mean(male_sentiment == 2)
female_positive_ratio = np.mean(female_sentiment == 2)

print(f"Before Calibration - Male Sentences - Positive Sentiment Ratio: {male_positive_ratio}")
print(f"Before Calibration - Female Sentences - Positive Sentiment Ratio: {female_positive_ratio}")

# calibrate predictions to balance positive sentiment ratios (changing some neutral to positive)
# i am only adjusting a fraction based on the difference between the sentiment ratios for male/female
if male_positive_ratio > female_positive_ratio:
    female_neutral_indices = np.where(y_pred_sentiment[female_indices] == 1)[0]
    num_to_adjust = int(len(female_neutral_indices) * (male_positive_ratio - female_positive_ratio))
    y_pred_sentiment[female_indices[female_neutral_indices[:num_to_adjust]]] = 2
elif female_positive_ratio > male_positive_ratio:
    male_neutral_indices = np.where(y_pred_sentiment[male_indices] == 1)[0]
    num_to_adjust = int(len(male_neutral_indices) * (female_positive_ratio - male_positive_ratio))
    y_pred_sentiment[male_indices[male_neutral_indices[:num_to_adjust]]] = 2 

# re-evaluate after calibration
accuracy_calibrated = accuracy_score(y_test_sentiment, y_pred_sentiment)
f1_calibrated = f1_score(y_test_sentiment, y_pred_sentiment, average='weighted')
print(f"Calibrated Sentiment Analysis - Accuracy: {accuracy_calibrated}, F1-Score: {f1_calibrated}")

# get new positive sentiment ratios
male_positive_ratio_calibrated = np.mean(y_pred_sentiment[male_indices] == 2)
female_positive_ratio_calibrated = np.mean(y_pred_sentiment[female_indices] == 2)

print(f"After Calibration - Male Sentences - Positive Sentiment Ratio: {male_positive_ratio_calibrated}")
print(f"After Calibration - Female Sentences - Positive Sentiment Ratio: {female_positive_ratio_calibrated}")