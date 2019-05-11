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

PICKLES=['sentiment_vectors.pickle', 'sentiment_labels.pickle']

def load_imdb_data():
	(X_train, y_train), (X_test, y_test)=imdb.load_data()
	X=np.concatenate((X_train, X_test), axis=0)
	y=np.concatenate((y_train, y_test), axis=0)
	X=sequence.pad_sequences(X, maxlen=400)

	X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
	X_train=np.resize(X_train, (X_train.shape[0], 1, X_train.shape[1]))
	X_test=np.resize(X_test, (X_test.shape[0], 1, X_test.shape[1]))
	return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

def load_sentiment_data():
	sentiment_x=pickle.load(open(PICKLES[0], 'rb'))
	sentiment_y=pickle.load(open(PICKLES[1], 'rb'))

	X_train, X_test, y_train, y_test=train_test_split(sentiment_x, sentiment_y, test_size=0.2)
	X_train=np.asarray(X_train)
	X_test=np.asarray(X_test)
	X_train=np.resize(X_train, (X_train.shape[0], 1, X_train.shape[2]))
	X_test=np.resize(X_test, (X_test.shape[0], 1, X_test.shape[2]))
	return X_train, X_test, np.asarray(y_train), np.asarray(y_test)

def encode_one_hot(labels):
	enc_labels=to_categorical(labels)
	# print(enc_labels.shape)
	return enc_labels

def build_model(training_imdb_x, training_imdb_y, training_sentiment_x, training_sentiment_y):
	input_sentiment=Input(shape=(training_sentiment_x.shape[1], training_sentiment_x.shape[2], ))
	input_imdb=Input(shape=(training_imdb_x.shape[1], training_imdb_x.shape[2], ))

	# lstm_sentiment1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True))(input_sentiment)
	# lstm_imdb1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True))(input_imdb)

	# lstm_sentiment1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True)(input_sentiment)
	# lstm_imdb1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True)(input_imdb)

	# lstm_sentiment2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment2', return_sequences=True))(lstm_sentiment1)
	# lstm_imdb2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb2', return_sequences=True))(lstm_imdb1)

	dense_sentiment1=Dense(256, activation='relu', name='dense_sentiment1')(input_sentiment)
	dropout_sentiment1=Dropout(0.5)(dense_sentiment1)
	dense_sentiment2=Dense(512, activation='relu', name='dense_sentiment2')(dropout_sentiment1)

	dense_imdb1=Dense(256, activation='relu', name='dense_imdb1')(input_imdb)
	dropout_imdb1=Dropout(0.5)(dense_imdb1)
	dense_imdb2=Dense(512, activation='relu', name='dense_imdb2')(dropout_imdb1)


	flat_sentiment=dense_sentiment2
	flat_imdb=dense_imdb2

	concat=concatenate([flat_sentiment, flat_imdb])
	# lstm_shared1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1'))(concat)
	# lstm_shared1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1')(concat)
	# lstm_shared2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared2'))(lstm_shared1)
	dense_shared1=Dense(128, activation='relu', name='dense_shared1')(concat)
	dropout_shared1=Dropout(0.5)(dense_shared1)
	dense_shared2=Dense(256, activation='relu', name='dense_shared2')(dropout_shared1)
	dropout_shared2=Dropout(0.5)(dense_shared2)
	dense_shared3=Dense(512, activation='relu', name='dense_shared3')(dropout_shared2)
	# dropout1=Dropout(0.5)(dense_shared1)
	# dense_shared2=Dense(256, activation='relu')(dropout1)
	dropout2=dense_shared3

	dropout2=Flatten()(dropout2)

	dense_sentiment_out1=Dense(2, activation='softmax')(dropout2)
	dense_imdb_out1=Dense(2, activation='softmax')(dropout2)

	model=Model([input_sentiment, input_imdb], [dense_sentiment_out1, dense_imdb_out1])
	model.summary()
	return model

training_imdb_x, test_imdb_x, training_imdb_y, test_imdb_y=load_imdb_data()
training_sentiment_x, test_sentiment_x, training_sentiment_y, test_sentiment_y=load_sentiment_data()

# print(training_imdb_x.shape, training_sentiment_x.shape)
# print(test_imdb_x.shape, test_sentiment_x.shape)

training_imdb_y=encode_one_hot(training_imdb_y)
training_sentiment_y=encode_one_hot(training_sentiment_y)
test_imdb_y=encode_one_hot(test_imdb_y)
test_sentiment_y=encode_one_hot(test_sentiment_y)


ind=1

for batch in range(1, 14):
	if(batch==13):
		pass
		print(len(training_imdb_x[(ind-1)*len(training_sentiment_x):]))
		print(len(training_sentiment_x[:2116]))
		print((ind-1)*len(training_sentiment_x))

		pass_training_x=[training_sentiment_x[:2116], training_imdb_x[(ind-1)*len(training_sentiment_x):]]
		pass_training_y=[training_sentiment_y[:2116], training_imdb_y[(ind-1)*len(training_sentiment_y):]]
	else:
		print(len(training_imdb_x[(ind-1)*len(training_sentiment_x):ind*len(training_sentiment_x)]))
		print(len(training_sentiment_x))
		print((ind-1)*len(training_sentiment_x), ind*len(training_sentiment_x))
		
		pass_training_x=[training_sentiment_x, training_imdb_x[(ind-1)*len(training_sentiment_x):ind*len(training_sentiment_x)]]
		pass_training_y=[training_sentiment_y, training_imdb_y[(ind-1)*len(training_sentiment_y):ind*len(training_sentiment_y)]]

	ind+=1

	if(batch==1):
		model=build_model(training_imdb_x, training_imdb_y, training_sentiment_x, training_sentiment_y)
	model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], loss_weights = [1.0, 0.5], metrics = ['accuracy'])
	model.fit(pass_training_x, pass_training_y, epochs=3, batch_size=64, shuffle=True, verbose=2, validation_split=0.1)
	l=model.evaluate([test_sentiment_x, test_imdb_x[:len(test_sentiment_x)]], [test_sentiment_y, test_imdb_y[:len(test_sentiment_y)]], batch_size=64, verbose=1)
	print(l)
	print(model.metrics_names)

	if(batch==13):
		model_json=model.to_json()
		with open("checkpoints_dense/model_final.json", "w") as json_file:
		    json_file.write(model_json)

		model.save_weights("checkpoints_dense/model_final.h5")
		print("Saved model_final to disk")
	else:
		model_json=model.to_json()
		with open("checkpoints_dense/model_"+str(batch)+".json", "w") as json_file:
		    json_file.write(model_json)

		model.save_weights("checkpoints_dense/model"+str(batch)+".h5")
		print("Saved model "+str(batch)+" to disk")

	if(batch==13):
		pass
	else:
		json_file=open('checkpoints_dense/model_'+str(batch)+'.json', 'r')
		loaded_model_json=json_file.read()
		json_file.close()
		model=model_from_json(loaded_model_json)

		model.load_weights("checkpoints_dense/model"+str(batch)+".h5")
		print("Loaded model "+str(batch)+" from disk") 

		for layer in model.layers[:-1]:
			layer.trainable=False

		for layer in model.layers:
			print(layer, layer.trainable)


	# 2 lstms, 1 shared lstm = [0.92094258688673192, 0.57308388646644881, 0.69571740883815136, 0.74177215250232553, 0.46582278511192227]
	# 2 lstms, 1 shared bidirectional lstm = [0.92357979484751251, 0.57663772513594813, 0.69388414425185962, 0.74177215250232553, 0.51139240430880195]
	# 2 bidirectional lstms, 1 shared bidirectional lstm = [0.92221749402299713, 0.57479483190971081, 0.69484532105771801, 0.74177215250232553, 0.51898734147035619]
	# 4 bidirectional lstms, 1 shared bidirectional lstm = [0.92038030835646623, 0.57294997399366354, 0.69486064941068237, 0.74177215250232553, 0.48354430304297918]
	# 4 bidirectional lstms, 2 shared bidirectional lstms = [0.91904836817632751, 0.573378893846198, 0.6913389551488659, 0.74177215250232553, 0.52151898794536344]
	# 4 bidirectional lstms, 2 shared bidirectional lstms, 2 shared dense = [0.91933497477181347, 0.572610114043272, 0.69344973111454444, 0.74177215250232553, 0.52531645614889599]
