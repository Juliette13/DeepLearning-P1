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
    - embedding layer : Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for
        Word Representation. (https://nlp.stanford.edu/projects/glove/)
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten
from keras import metrics
from keras.utils import plot_model


EMBEDDING_DIM = 50  # 50, 100, 200 ou 300
MAX_SEQUENCE_LENGTH = 150
TRAIN = True  # True pour entrainer le modèle, False pour juste le charger


def padding_data(sequences):
    """
    Ajout de zéros pour uniformiser la taille des listes
    :param sequences: listes de données
    :param max_length: taille que doit faire la liste
    :return: données uniformisées
    """
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return data


def load_embedding_model():
    """
    Chargement du modèle pré entrainée GloVe
    """
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.'+str(EMBEDDING_DIM)+'d.txt', encoding="utf8")
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
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Embedding matrix ready')
    return embedding_matrix


def build_model(max_fatures, dim_embedding, embedding_matrix):
    """
    Création du modèle
    :param input_dim: Taille du vecteurs d'entrée
    :param output_dim: Taille du vecteurs de sortie
    :param embedding_matrix: Matrice de plongement de mots
    :return: Structure du modèle
    """
    model = Sequential()
    model.add(Embedding(max_fatures, dim_embedding, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
    model.add(LSTM(20, dropout=0.5, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    return model


def save_model_weights(model):
    """
    :param model:
    :return:
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def plot_fit_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('proper_df.csv', index_col=0)
    print(df.head())

    X = df['text_lemmatizer']
    y = df['polarity']
    y = y.map(lambda x: 1 if x == 4 else x)
    y = pd.get_dummies(y).values
    print(y)

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
    X = tokenizer.texts_to_sequences(X)

    # Uniformisation de la taille du tableau. /!\ >> A voir avec la moyenne/médiane de la longueur des tweets.
    print(len(X[0]))
    X = padding_data(X)
    print(len(X[0]))

    # Separation du dataset en train, test, validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    if TRAIN:
        embedding_matrix = get_embedding_matrix(vocab_size, tokenizer)

        model = build_model(max_fatures=vocab_size, dim_embedding=EMBEDDING_DIM, embedding_matrix=embedding_matrix)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics.binary_accuracy])
        print(model.summary())
        try:
            # Download GraphViz to save PNG (https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi)
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        except:
            print("Can't save the model as an image")
        history = model.fit(X_train, y_train, epochs=5, verbose=1)
        plot_fit_history(history)
        model.save("model.h5")
        print("Model saved")
    else:
        model = load_model('model.h5')
        model.summary()
        print("Model loaded")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % (loss * 100))
