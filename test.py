import matplotlib
matplotlib.use('Agg')

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
import matplotlib.pyplot as plt

PICKLES=['sentiment_vectors.pickle', 'sentiment_labels.pickle', 'sentiment_embedding_vectors.pickle']
VOCAB_SIZE=16864
PLOTS_FOLDER='PLOTS_FOLDER/'

# def load_imdb_data():
# 	(X_train, y_train), (X_test, y_test)=imdb.load_data()
# 	X=np.concatenate((X_train, X_test), axis=0)
# 	y=np.concatenate((y_train, y_test), axis=0)
# 	X=sequence.pad_sequences(X, maxlen=400)

# 	X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
# 	X_train=np.resize(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# 	X_test=np.resize(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# 	return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)

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

# def load_sentiment_data():
# 	sentiment_x=pickle.load(open(PICKLES[0], 'rb'))
# 	sentiment_y=pickle.load(open(PICKLES[1], 'rb'))

# 	X_train, X_test, y_train, y_test=train_test_split(sentiment_x, sentiment_y, test_size=0.2)
# 	X_train=np.asarray(X_train)
# 	X_test=np.asarray(X_test)
# 	X_train=np.resize(X_train, (X_train.shape[0], 1, X_train.shape[2]))
# 	X_test=np.resize(X_test, (X_test.shape[0], 1, X_test.shape[2]))
# 	return X_train, X_test, np.asarray(y_train), np.asarray(y_test)

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


def plot_required(history):
	# dict_keys(['val_dense_1_acc', 'loss', 'val_dense_2_acc', 'dense_2_loss', 'val_dense_2_loss', 'dense_1_loss', 'val_dense_1_loss', 'val_loss', 'dense_2_acc', 'dense_1_acc'])

	n_epochs=len(history.history['loss'])+1
	
	#dense_1_acc
	plt.figure(1)
	plt.plot(np.arange(1, n_epochs), history.history['dense_1_acc'])
	plt.xlabel('Epochs')
	plt.ylabel('Sentiment accuracy')
	plt.savefig(PLOTS_FOLDER+'dense_1_acc.png')
	# plt.show()

	#dense_1_loss
	plt.figure(2)
	plt.plot(np.arange(1, n_epochs), history.history['dense_1_loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Sentiment training loss')
	plt.savefig(PLOTS_FOLDER+'dense_1_loss.png')
	# plt.show()

	#dense_3_acc
	plt.figure(3)
	plt.plot(np.arange(1, n_epochs), history.history['dense_2_acc'])
	plt.xlabel('Epochs')
	plt.ylabel('IMDB accuracy')
	plt.savefig(PLOTS_FOLDER+'dense_2_acc.png')
	# plt.show()

	#dense_3_loss
	plt.figure(4)
	plt.plot(np.arange(1, n_epochs), history.history['dense_2_loss'])
	plt.xlabel('Epochs')
	plt.ylabel('IMDB training loss')
	plt.savefig(PLOTS_FOLDER+'dense_2_loss.png')
	# plt.show()

	#loss
	plt.figure(5)
	plt.plot(np.arange(1, n_epochs), history.history['loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Training loss')
	plt.savefig(PLOTS_FOLDER+'loss.png')
	# plt.show()

	plt.figure(6)
	plt.plot(np.arange(1, n_epochs), history.history['val_dense_1_acc'])
	plt.xlabel('Epochs')
	plt.ylabel('Sentiment val accuracy')
	plt.savefig(PLOTS_FOLDER+'sentiment_val_acc.png')

	plt.figure(7)
	plt.plot(np.arange(1, n_epochs), history.history['val_dense_2_acc'])
	plt.xlabel('Epochs')
	plt.ylabel('IMDB val accuracy')
	plt.savefig(PLOTS_FOLDER+'IMDB_val_acc.png')

	plt.figure(8)
	plt.plot(np.arange(1, n_epochs), history.history['val_loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Validation loss')
	plt.savefig(PLOTS_FOLDER+'val_loss.png')

	plt.figure(9)
	plt.plot(np.arange(1, n_epochs), history.history['val_dense_2_loss'])
	plt.xlabel('Epochs')
	plt.ylabel('IMDB val loss')
	plt.savefig(PLOTS_FOLDER+'IMDB_val_loss.png')

	plt.figure(10)
	plt.plot(np.arange(1, n_epochs), history.history['val_dense_1_loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Sentiment val loss')
	plt.savefig(PLOTS_FOLDER+'Sentiment_val_loss.png')

def build_model(training_imdb_x, training_imdb_y, training_sentiment_x, training_sentiment_y):
	input_sentiment=Input(shape=(training_sentiment_x.shape[1], ))
	input_imdb=Input(shape=(training_imdb_x.shape[1], ))
	
	emb_imdb=Embedding(10000, 32, input_length=400)(input_imdb)
	emb_sentiment=Embedding(VOCAB_SIZE, 32, input_length=400)(input_sentiment)

	lstm_imdb=LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(emb_imdb)
	lstm_sentiment=LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(emb_sentiment)

	lstm_imdb1=LSTM(100, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)(lstm_imdb)
	lstm_sentiment1=LSTM(100, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)(lstm_sentiment)

	# flatten_imdb=Flatten()(lstm_imdb1)
	# flatten_sentiment=Flatten()(lstm_sentiment1)

	flatten_imdb=lstm_imdb1
	flatten_sentiment=lstm_sentiment1

	concat=concatenate([flatten_imdb, flatten_sentiment])

	lstm_shared=Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(concat)

	out=Flatten()(lstm_shared)

	dense_imdb=Dense(1, activation='sigmoid')(out)
	dense_sentiment=Dense(1, activation='sigmoid')(out)

	# lstm_sentiment1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True))(input_sentiment)
	# lstm_imdb1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True))(input_imdb)

	# lstm_sentiment1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment1', return_sequences=True)(input_sentiment)
	# lstm_imdb1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb1', return_sequences=True)(input_imdb)

	# # lstm_sentiment2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_sentiment2', return_sequences=True))(lstm_sentiment1)
	# # lstm_imdb2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_imdb2', return_sequences=True))(lstm_imdb1)

	# flat_sentiment=lstm_sentiment1
	# flat_imdb=lstm_imdb1

	# concat=concatenate([flat_sentiment, flat_imdb])
	# # lstm_shared1=Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1'))(concat)
	# lstm_shared1=LSTM(128, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared1')(concat)
	# # lstm_shared2=Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, name='lstm_shared2'))(lstm_shared1)
	# dense_shared1=Dense(128, activation='relu')(lstm_shared1)
	# # dropout1=Dropout(0.5)(dense_shared1)
	# # dense_shared2=Dense(256, activation='relu')(dropout1)
	# dropout2=dense_shared1

	# dense_sentiment1=Dense(2, activation='softmax')(dropout2)
	# dense_imdb1=Dense(2, activation='softmax')(dropout2)

	model=Model([input_sentiment, input_imdb], [dense_sentiment, dense_imdb])
	model.summary()
	return model

training_imdb_x, test_imdb_x, training_imdb_y, test_imdb_y=load_imdb_data()
training_sentiment_x, test_sentiment_x, training_sentiment_y, test_sentiment_y=load_sentiment_data()

print(training_imdb_x.shape, training_sentiment_x.shape)
print(test_imdb_x.shape, test_sentiment_x.shape)

# training_imdb_y=encode_one_hot(training_imdb_y)
# training_sentiment_y=encode_one_hot(training_sentiment_y)
# test_imdb_y=encode_one_hot(test_imdb_y)
# test_sentiment_y=encode_one_hot(test_sentiment_y)


ind=1

for batch in range(1, 9):
	if(batch==8):
		pass
		print(len(training_imdb_x[(ind-1)*len(training_sentiment_x):]))
		print(len(training_sentiment_x[:2901]))
		print((ind-1)*len(training_sentiment_x))

		pass_training_x=[training_sentiment_x[:2901], training_imdb_x[(ind-1)*len(training_sentiment_x):]]
		pass_training_y=[training_sentiment_y[:2901], training_imdb_y[(ind-1)*len(training_sentiment_y):]]
	else:
		print(len(training_imdb_x[(ind-1)*len(training_sentiment_x):ind*len(training_sentiment_x)]))
		print(len(training_sentiment_x))
		print((ind-1)*len(training_sentiment_x), ind*len(training_sentiment_x), ind)
		
		pass_training_x=[training_sentiment_x, training_imdb_x[(ind-1)*len(training_sentiment_x):ind*len(training_sentiment_x)]]
		pass_training_y=[training_sentiment_y, training_imdb_y[(ind-1)*len(training_sentiment_y):ind*len(training_sentiment_y)]]

	ind+=1

	if(batch==1):
		model=build_model(training_imdb_x, training_imdb_y, training_sentiment_x, training_sentiment_y)
	model.compile(optimizer='adam', loss=['binary_crossentropy', 'binary_crossentropy'], loss_weights = [1.0, 0.5], metrics = ['accuracy'])
	history=model.fit(pass_training_x, pass_training_y, epochs=10, batch_size=64, shuffle=True, verbose=2, validation_split=0.1)
	l=model.evaluate([test_sentiment_x, test_imdb_x[:len(test_sentiment_x)]], [test_sentiment_y, test_imdb_y[:len(test_sentiment_y)]], batch_size=64, verbose=1)
	

	print(history.history.keys())
	plot_required(history)

	print(l)
	print(model.metrics_names)

	if(batch==8):
		model_json=model.to_json()
		with open("checkpoints_lstm/model_final.json", "w") as json_file:
		    json_file.write(model_json)

		model.save_weights("checkpoints_lstm/model_final.h5")
		print("Saved model_final to disk")
	else:
		model_json=model.to_json()
		with open("checkpoints_lstm/model_"+str(batch)+".json", "w") as json_file:
		    json_file.write(model_json)

		model.save_weights("checkpoints_lstm/model"+str(batch)+".h5")
		print("Saved model "+str(batch)+" to disk")

	if(batch==8):
		pass
	else:
		json_file=open('checkpoints_lstm/model_'+str(batch)+'.json', 'r')
		loaded_model_json=json_file.read()
		json_file.close()
		model=model_from_json(loaded_model_json)

		model.load_weights("checkpoints_lstm/model"+str(batch)+".h5")
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
