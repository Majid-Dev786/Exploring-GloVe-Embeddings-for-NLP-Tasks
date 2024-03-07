# Importing necessary libraries from NumPy and TensorFlow Keras
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Sample sentences to process
sentences = [
    'I love coding',
    'Python is my favorite programming language',
    'Natural Language Processing is interesting'
]

# Corresponding labels for the sentences indicating categories
labels = [0, 1, 1]

# Initializing the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # Learning the word index
vocab_size = len(tokenizer.word_index) + 1 # Vocabulary size
sequences = tokenizer.texts_to_sequences(sentences) # Converting texts to sequences

# Determining the maximum sequence length
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length) # Padding sequences for uniform length

# GloVe embeddings file setup
glove_file = 'glove.6B.100d.txt'
embedding_dim = 100
embeddings_index = {} # Dictionary to store word embeddings
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs # Storing word embeddings

# Preparing the embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector # Assigning embeddings to matrix

categorical_labels = to_categorical(labels) # Converting labels to categorical format

# Model definition
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                    input_length=max_sequence_length, trainable=False)) # Adding embedding layer
model.add(LSTM(128)) # Adding LSTM layer
model.add(Dense(2, activation='softmax')) # Output layer

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, categorical_labels, epochs=10) # Training the model

# Preparing test sentence
test_sentence = 'I enjoy programming'
test_sequence = tokenizer.texts_to_sequences([test_sentence]) # Converting to sequence
padded_test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length) # Padding the sequence

# Predicting sentiment
predicted_probabilities = model.predict(padded_test_sequence)[0]
predicted_label = np.argmax(predicted_probabilities)
sentiment = 'Positive' if predicted_label == 1 else 'Negative'
print(f'Sentiment prediction for "{test_sentence}": {sentiment}') # Printing the prediction
