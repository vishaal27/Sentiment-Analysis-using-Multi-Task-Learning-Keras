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

def load_imdb_data():
	(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=10000)
	# X=np.concatenate((X_train, X_test), axis=0)
	# y=np.concatenate((y_train, y_test), axis=0)
	# X=sequence.pad_sequences(X, maxlen=400)

	X_train=sequence.pad_sequences(X_train, maxlen=400)
	X_test=sequence.pad_sequences(X_test, maxlen=400)

	# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
	# X_train=np.resize(X_train, (X_train.shape[0], 1, X_train.shape[1]))
	# X_test=np.resize(X_test, (X_test.shape[0], 1, X_test.shape[1]))
	return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

def encode_one_hot(labels):
	enc_labels=to_categorical(labels)
	# print(enc_labels.shape)
	return enc_labels

def build_model(training_imdb_x, training_imdb_y):
	# input_sentiment=Input(shape=(training_imdb_x.shape[1], training_imdb_x.shape[2], ))

	# lstm_sentiment1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True))(input_sentiment)
	# lstm_imdb1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True))(input_imdb)

	# lstm_sentiment1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True)(input_sentiment)
	# lstm_imdb1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True)(input_imdb)

	# lstm_sentiment2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment2', return_sequences=True))(lstm_sentiment1)
	# lstm_imdb2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb2', return_sequences=True))(lstm_imdb1)
	
	model=Sequential()
	model.add(Embedding(10000, 32, input_length=400))
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



	# embedding1=Embedding(5000, 32, input_length=400)
	# embedding1=Flatten()(embedding1)
	# dense_sentiment1=Dense(32, activation='relu', name='dense_sentiment1')(embedding1)
	# dropout_sentiment1=Dropout(0.5)(dense_sentiment1)
	# dense_sentiment2=Dense(64, activation='relu', name='dense_sentiment2')(dropout_sentiment1)

	# flat_sentiment=dense_sentiment2

	# # lstm_shared1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1'))(concat)
	# # lstm_shared1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1')(concat)
	# # lstm_shared2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared2'))(lstm_shared1)
	# dense_sentiment3=Dense(128, activation='relu', name='dense_sentiment3')(flat_sentiment)
	# dropout_sentiment2=Dropout(0.5)(dense_sentiment3)
	# dense_sentiment4=Dense(256, activation='relu', name='dense_sentiment4')(dropout_sentiment2)
	# dropout_sentiment3=Dropout(0.5)(dense_sentiment4)
	# dense_sentiment5=Dense(512, activation='relu', name='dense_sentiment5')(dropout_sentiment3)
	# # dropout1=Dropout(0.5)(dense_shared1)
	# # dense_shared2=Dense(256, activation='relu')(dropout1)
	# dropout2=dense_sentiment5

	# dropout2=Flatten()(dropout2)

	# dense_sentiment_out1=Dense(1, activation='softmax')(dropout2)

	# model=Model(input_sentiment, dense_sentiment_out1)
	model.summary()
	return model

training_imdb_x, test_imdb_x, training_imdb_y, test_imdb_y=load_imdb_data()

print(training_imdb_x.shape, training_imdb_y.shape)
# training_imdb_y=encode_one_hot(training_imdb_y)
# test_imdb_y=encode_one_hot(test_imdb_y)
	
model=build_model(training_imdb_x, training_imdb_y)
model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics = ['accuracy'])
model.fit(training_imdb_x, training_imdb_y, epochs=25, batch_size=64, shuffle=True, verbose=2, validation_split=0.1)
l=model.evaluate(test_imdb_x, test_imdb_y, batch_size=64, verbose=1)
print(l)
print(model.metrics_names)

model_json=model.to_json()
with open("checkpoints_imdb/model_final.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("checkpoints_imdb/model_final.h5")
print("Saved model_final to disk")