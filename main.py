"""
Source:
    - Aller plus loin avec le preprocessing: https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    - Keras docs:
        - https://keras.io/preprocessing/text/#tokenizer
        - https://keras.io/preprocessing/sequence/#pad_sequences
        - https://keras.io/layers/embeddings/#embedding
        - https://keras.io/layers/recurrent/#lstm
        - https://keras.io/metrics/
        - https://keras.io/activations/
    - embedding layer : Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. (https://nlp.stanford.edu/projects/glove/)
"""

import pickle
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras import metrics


embedding_dim = 50# 50, 100, 200 ou 300
max_length = 150


def padding_data(sequences, max_length):
    """
    Ajout de zéros pour uniformiser la taille des listes
    :param sequences: listes de données
    :param max_length: taille que doit faire la liste
    :return: données uniformisées
    """
    data = pad_sequences(sequences, maxlen=max_length, padding='post')
    return data


def load_embedding_model():
    """
    Chargement du model pré entrainée GloVe
    """
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.'+str(embedding_dim)+'d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def get_embedding_matrix(vocab_size, tokenizer):
    """

    :param vocab_size: Taille du vocabulaire
    :param tokenizer: Tokenizer (déjà entrainé)
    :return: Matrice de plongement de mots
    """
    embeddings_index = load_embedding_model()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Embedding matrix ready')
    return embedding_matrix


def build_model(input_dim, output_dim, embedding_matrix):
    """
    Création du modèle
    :param input_dim: Taille du vecteurs d'entrée
    :param output_dim: Taille du vecteurs de sortie
    :param embedding_matrix: Matrice de plongement de mots
    :return: Structure du modèle
    """
    model = Sequential()
    model.add(Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    return model


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('proper_df.csv', index_col=0)
    print(df.head())

    X = df['text_lemmatizer']
    y = df['polarity']
    y = y.map(lambda x: 1 if x == 4 else x)

    # Separation du dataset en train, test, validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # Tokenization du texte
    if os.path.isfile('tokenizer.pickle'):
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print('Tokenizer loaded')
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        print('Tokenizer fit on texts')
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary size:', vocab_size)

    # Transformation des mots en nombres par le tokenizer
    X_train = tokenizer.texts_to_sequences(X_train)

    # Uniformisation de la taille du tableau. /!\ >> A voir avec la moyenne/médiane de la longueur des tweets.
    print(len(X_train[0]), X_train[0])
    X_train = padding_data(X_train, max_length)
    print(len(X_train[0]), X_train[0])

    print(X_train)

    embedding_matrix = get_embedding_matrix(vocab_size, tokenizer)
    print(embedding_matrix)

    model = build_model(input_dim=vocab_size, output_dim=embedding_dim, embedding_matrix=embedding_matrix)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics.binary_accuracy])
    print(model.summary())
    model.fit(X_train, y_train, epochs=5, verbose=1)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % (loss * 100))
