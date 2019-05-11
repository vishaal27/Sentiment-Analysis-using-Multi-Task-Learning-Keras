import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from gensim.models import KeyedVectors
import pickle
from keras.preprocessing.text import Tokenizer

DATA_PATH='data/'
train_set='train.csv'

PICKLES=['sentiment_vectors.pickle', 'sentiment_labels.pickle', 'sentiment_embedding_vectors.pickle']
MODEL_BIN='sentiment_word2vec.bin'

def create_word_vectors(sentiment_paras):
	if(os.path.isfile(MODEL_BIN)):
		model=Word2Vec.load(MODEL_BIN)
	else:
		all_sentences=[]
		for sent in sentiment_paras:
			all_sentences.append(word_tokenize(sent))

		model=Word2Vec(all_sentences, size=100, min_count=1, workers=10)
		model.save(MODEL_BIN)

	para_vectors=[]
	print(len(model.wv.vocab))
	iter=0
	for para in sentiment_paras:
		iter+=1
		vec=np.zeros((1, 100))
		for word in para:
			try:
				vec+=model.wv[word]	
			except Exception as e:
				pass
		print('Completed '+str(iter))
		para_vectors.append(vec/len(para))
	return para_vectors

def create_embedding_vectors(sentiment_paras):
	tokenizer=Tokenizer(num_words=10000)
	tokenizer.fit_on_texts(sentiment_paras)
	vectors=tokenizer.texts_to_sequences(sentiment_paras)
	print("vocab size: "+str(len(tokenizer.word_index)+1))
	return vectors

df=pd.read_csv(DATA_PATH+train_set)

sentiment_data=df.values
sentiment_labels=sentiment_data[:, 0]
sentiment_paras=sentiment_data[:, 2]

print(sentiment_labels.shape, sentiment_paras.shape)

word_vectors=create_word_vectors(sentiment_paras)
embedding_vectors=create_embedding_vectors(sentiment_paras)

pickle.dump(word_vectors, open(PICKLES[0], 'wb'))
pickle.dump(sentiment_labels, open(PICKLES[1], 'wb'))
pickle.dump(embedding_vectors, open(PICKLES[2], 'wb'))