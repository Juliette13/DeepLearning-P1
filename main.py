"""
Matrice confusion : [[131837  28319]
                    [ 29520 130324]]
Accuracy : 0.819253
Recall : 0.815320
Precision : 0.821492
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout, LSTM


SEQUENCE_LENGTH = 150  # Taille maximale d'une phrase
EMBEDDING_DIM = 200  # Dimension de la matrice d'embedding GloVe
BATCH_SIZE = 32  # Batch size du fir


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('proper_df.csv', index_col=0)
    print(df.head())

    X = df['text']
    X = X.astype(str)

    y = df['polarity']
    y = y.map(lambda x: 1 if x == 4 else x)
    y = pd.get_dummies(y).values  # Equivalent à la fonction one_hot de Keras

    print(X.shape)
    print(y.shape)

    # Entrainement du tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    vocab_size = len(tokenizer.word_index) + 1

    # Remplace le texte par une séquence d'entiers
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH, padding='post')

    # Chargement de la matrice d'embeddings
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.'+str(EMBEDDING_DIM)+'d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # Initialisation de la matrice d'embeddings avec les mots du dictionnaires (présent dans le tokenizer)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Split des dataframes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # Model
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False))
    model.add(LSTM(50, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True))
    # Return_sequence True renvoie 3D donc activer Flatten, sinon renvoie 2D donc commenter Flatten
    model.add(Flatten())
    model.add(Dropout(0.5))  # Réduire l'overfitting
    model.add(Dense(2, activation='sigmoid'))  # 1 sortie par classes
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=15, verbose=1, validation_data=(X_val, y_val),
                        shuffle=True)
    model.save("model.h5")

    try:
        # Graphique d'évlution de l'accuracy par epochs
        plt.subplot(1)
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend(loc='upper left')
        plt.savefig('accuracy.png')
        plt.clf()
        # Graphique d'évlution de la perte par epochs
        plt.subplot(2)
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend(loc='upper left')
        plt.savefig('loss.png')
        plt.clf()
    except:
        pass

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    tn, fp, fn, tp = conf_matrix.ravel()
    plt.clf()
    plt.imshow(conf_matrix, interpolation='nearest')
    classNames = ['Negative', 'Positive']
    plt.title('Classification Confusion Matrix on Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(conf_matrix[i][j]))
    plt.savefig('confusion matrix.png')

    # Métriques dévaluations du modèle
    print('Accuracy : %f' % ((tn + tp) / (tn + fp + fn + tp)))
    print('Recall : %f' % (tp / (fn + tp)))
    print('Precision : %f' % (tp / (fp + tp)))
