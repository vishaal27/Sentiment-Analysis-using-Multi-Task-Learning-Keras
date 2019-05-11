import numpy as np
from keras.datasets import imdb
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding
import keras
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import model_from_json

PICKLES=['sentiment_vectors.pickle', 'sentiment_labels.pickle', 'sentiment_embedding_vectors.pickle']
VOCAB_SIZE=16864

def load_sentiment_data():
	# sentiment_x=pickle.load(open(PICKLES[0], 'rb'))
	sentiment_x=pickle.load(open(PICKLES[2], 'rb'))
	sentiment_y=pickle.load(open(PICKLES[1], 'rb'))

	X_train, X_test, y_train, y_test=train_test_split(sentiment_x, sentiment_y, test_size=0.2)
	X_train=np.asarray(X_train)
	X_test=np.asarray(X_test)
	X_train=sequence.pad_sequences(X_train, maxlen=400)
	X_test=sequence.pad_sequences(X_test, maxlen=400)
	# X_train=np.resize(X_train, (X_train.shape[0], X_train.shape[2]))
	# X_test=np.resize(X_test, (X_test.shape[0], X_test.shape[2]))
	return X_train, X_test, np.asarray(y_train), np.asarray(y_test)

def encode_one_hot(labels):
	enc_labels=to_categorical(labels)
	# print(enc_labels.shape)
	return enc_labels

def build_model(training_sentiment_x, training_sentiment_y):
	# input_sentiment=Input(shape=(training_sentiment_x.shape[1], training_sentiment_x.shape[2], ))

	# lstm_sentiment1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True))(input_sentiment)
	# lstm_imdb1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True))(input_imdb)

	# lstm_sentiment1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True)(input_sentiment)
	# lstm_imdb1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True)(input_imdb)

	# lstm_sentiment2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment2', return_sequences=True))(lstm_sentiment1)
	# lstm_imdb2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb2', return_sequences=True))(lstm_imdb1)

	# print(lstm_sentiment1.shape)

	# lstm_sentiment1=Flatten()(lstm_sentiment1)

	# dense_sentiment1=Dense(32, activation='relu', name='dense_sentiment1')(lstm_sentiment1)
	# dropout_sentiment1=Dropout(0.5)(dense_sentiment1)
	# dense_sentiment2=Dense(64, activation='relu', name='dense_sentiment2')(dropout_sentiment1)

	# flat_sentiment=dense_sentiment2

	# lstm_shared1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1'))(concat)
	# lstm_shared1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1')(concat)
	# lstm_shared2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared2'))(lstm_shared1)
	# dense_sentiment3=Dense(128, activation='relu', name='dense_sentiment3')(flat_sentiment)
	# dropout_sentiment2=Dropout(0.5)(dense_sentiment3)
	# dense_sentiment4=Dense(256, activation='relu', name='dense_sentiment4')(dropout_sentiment2)
	# dropout_sentiment3=Dropout(0.5)(dense_sentiment4)
	# dense_sentiment5=Dense(512, activation='relu', name='dense_sentiment5')(dropout_sentiment3)
	# dropout1=Dropout(0.5)(dense_shared1)
	# dense_shared2=Dense(256, activation='relu')(dropout1)
	# dropout2=dense_sentiment5

	# dropout2=Flatten()(dropout2)

	# dense_sentiment_out1=Dense(2, activation='softmax')(dropout2)

	# model=Model(input_sentiment, dense_sentiment_out1)
	model=Sequential()
	model.add(Embedding(VOCAB_SIZE, 32, input_length=400))
	# model.add(Flatten())
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
	# model.add(Dense(32, activation='relu', name='dense_sentiment1'))
	# model.add(Dropout(0.5))
	# model.add(Dense(64, activation='relu', name='dense_sentiment2'))
	# model.add(Dense(128, activation='relu', name='dense_sentiment3'))
	# model.add(Dropout(0.5))
	# model.add(Dense(256, activation='relu', name='dense_sentiment4'))
	# model.add(Dropout(0.5))
	# model.add(Dense(512, activation='relu', name='dense_sentiment5'))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.summary()
	return model

training_sentiment_x, test_sentiment_x, training_sentiment_y, test_sentiment_y=load_sentiment_data()

# training_sentiment_y=encode_one_hot(training_sentiment_y)
# test_sentiment_y=encode_one_hot(test_sentiment_y)
	
print(training_sentiment_x.shape, test_sentiment_x.shape)

model=build_model(training_sentiment_x, training_sentiment_y)
model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics = ['accuracy'])
model.fit(training_sentiment_x, training_sentiment_y, epochs=50, batch_size=64, shuffle=True, verbose=2, validation_split=0.1)
l=model.evaluate(test_sentiment_x, test_sentiment_y, batch_size=64, verbose=1)
print(l)
print(model.metrics_names)

model_json=model.to_json()
with open("checkpoints_sentiment/model_final.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("checkpoints_sentiment/model_final.h5")
print("Saved model_final to disk")