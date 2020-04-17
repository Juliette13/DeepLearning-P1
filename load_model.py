import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout, LSTM
from keras.models import load_model


pd.set_option('display.max_columns', None)
df = pd.read_csv('proper_df.csv', index_col=0)

X = df['text']
X = X.astype(str)

y = df['polarity']
y = y.map(lambda x: 1 if x == 4 else x)
y = pd.get_dummies(y).values

t = Tokenizer()
t.fit_on_texts(X)
vocab_size = len(t.word_index) + 1
# integer encode the documents
X = t.texts_to_sequences(X)
# pad documents to a max length of 4 words
max_length = 150
X = pad_sequences(X, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

loaded_model = load_model('model.h5')
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %f' % (accuracy * 100))
print('Loss: %f' % (loss * 100))