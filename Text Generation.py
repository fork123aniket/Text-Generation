import string
from numpy import array
from pickle import dump, load
import numpy as np
from numpy import asarray, zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from random import randint
from tensorflow.keras.preprocessing.sequence import pad_sequences


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens


# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


# load document
in_filename = '/content/drive/MyDrive/Colab Notebooks/republic_clean.txt'
doc = load_doc(in_filename)
print(doc[:200])

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

mem_cell, drop, num_layers, pre_trained = [100, 100, 100], [0, 0, 0], 3, False
batch_size, n_epochs, n_words = 128, 100, 50
assert len(mem_cell) == num_layers and len(drop) == num_layers

# organize into sequences of tokens
length = n_words + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = '/content/drive/MyDrive/Colab Notebooks/republic_sequences.txt'
save_doc(sequences, out_filename)


# load
in_filename = '/content/drive/MyDrive/Colab Notebooks/republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


if pre_trained:
	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('/content/drive/MyDrive/Colab Notebooks/glove.6B.50d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))
	# create a weight matrix for words in training docs
	embedding_matrix = zeros((vocab_size, n_words))
	for word, i in tokenizer.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			embedding_matrix[i] = embeddings_index.get('unk')


# define model
def build_model(mem_cell, drop, num_layers, pre_trained=False):
	model = Sequential()
	if not pre_trained:
		model.add(Embedding(vocab_size, n_words, input_length=seq_length))
	else:
		model.add(Embedding(vocab_size, n_words, weights=[embedding_matrix],
							input_length=seq_length, trainable=False))
	for n_layer in range(num_layers):
		model.add(LSTM(mem_cell[n_layer], return_sequences=True))
		model.add(Dropout(drop[n_layer]))
		if n_layer == num_layers - 1:
			model.add(LSTM(mem_cell[n_layer]))
			model.add(Dropout(drop[n_layer]))
	model.add(Dense(mem_cell[-1], activation='relu'))
	model.add(Dropout(drop[-1]))
	model.add(Dense(vocab_size, activation='softmax'))
	return model


# compile model
model = build_model(mem_cell, drop, num_layers, pre_trained)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=batch_size, epochs=n_epochs)
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		predict_x = model.predict(encoded)
		yhat = np.argmax(predict_x, axis=1)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)


# load cleaned text sequences
in_filename = '/content/drive/MyDrive/Colab Notebooks/republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('/content/drive/MyDrive/Colab Notebooks/model.h5')

# load the tokenizer
tokenizer = load(open('/content/drive/MyDrive/Colab Notebooks/tokenizer.pkl', 'rb'))

# select a seed text
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')

# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, n_words)
print(generated)
