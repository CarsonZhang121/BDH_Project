from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import matplotlib.pyplot as plt
import re
import numpy as np
import random
import pandas as pd
import pickle
import os

os.environ['KERAS_BACKEND'] = 'theano'
STOPWORDS = stopwords.words('English')

# clean the raw text data
def preprocessor_cleantext(text):

    text = re.sub('\[\*\*[^\]]*\*\*\]','',text)             # remove all the information in [] inclusively
    text = re.sub('<[^>]*>','', text)                       # remove all the information in <> inclusively
    text = re.sub('[\W]+',' ',text.lower())                 # convert all the letters to lower case and replace non-word with space
    text = re.sub("\d+"," ", text)                          # replace all the digits with space

    return text

# create sequence for each text data
def createWordSequence(df, max_sequence_len = 600, inputCol = 'TEXT'):

    texts = df[inputCol].apply(preprocessor_cleantext)      # import and clean the text column
    toke = Tokenizer()                                      # chop the text into piceses and throwing away certain characters, such as punctuation
    toke.fit_on_texts(texts)                                # store all the words in index
    sequence = toke.texts_to_sequences(texts)               # convert the texts to sequence considering only the top num_words

    ave_seq = [len(i) for i in sequence]
    print ("Average text length is: {} ".format(1.0 * sum(ave_seq)/len(ave_seq)))

    word_index = toke.word_index                            # dictionary to store the {word: index}
    reverse_word_index = dict(zip(word_index.values(), word_index.keys()))    # reverse dictionary for easy search {index: word}
    print("Found {} unique tokens".format(len(word_index)))
    data = pad_sequences(sequence, maxlen = max_sequence_len)# trim the sequence to the length of max_sequence_len

    return data, word_index, reverse_word_index

# create embedding matrix for all the legal words in the texts
def create_EmbeddingMatrix (word_index, GloVe_model_path, remove_stopwords = True):

    glove_model = {}
    print("Loading GloVe Model")

    f = open(GloVe_model_path)
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        vector = np.asarray(splitLine[1:], dtype = 'float32')
        glove_model[word] = vector
    f.close()

    print("Found {} word vectors.".format(len(glove_model)))

    if remove_stopwords:
        keys_updated = [word for word in glove_model.keys() if word not in STOPWORDS]
        glove_model_set = set(keys_updated)
    else:
        glove_model_set = set(glove_model.keys())

    embedding_dim = len(glove_model["hello"])                   # dimension of the embedding vector
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word,i in word_index.items():                             # construct embedding matrix for all the legal words in the texts
        if word in glove_model_set:
            embedding_matrix[i] = glove_model.get(word)

    return embedding_matrix

# separate the samples into the group of train (60%), validation(15%) and test(25%)
def train_test_separator(seed, N):
    idx = list(range(N))
    random.seed(seed)
    random.shuffle(idx)
    idx_train = idx[0: int(N*0.50)]
    idx_val = idx[int(N*0.50):int(N*0.75)]
    idx_test = idx[int(N*0.75):N]

    return idx_train, idx_val, idx_test

# Output the pickle files
def output_pickle(obj, fname):
    f = open(fname, 'wb')
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL) # Note that cPickle is much faster than pickle
    f.close()

# LSTM Model
def LSTM_MODEL(input_shape, output_shape, embedding_layer):
    print("Building Model")
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape, name = 'input_layer'))
    model.add(embedding_layer)
    model.add(LSTM(256, return_sequences=True))           # returns a sequence of vectors of dimension 256
    model.add(Dropout(0.5))                               # Overcoming the overfitting problem
    model.add(BatchNormalization())                       # Speed up the learning speed
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_shape,activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc','mse'])
    model.summary()
    model.save_weights("../inputData/weights_LSTM.h5")
    return model

# Train Model
def train(reverse_word_index, embedding_matrix, train_data, train_label, val_data, val_label,
          test_data, test_label, nb_epoch = 50, batch_size = 128, pre_train = False):
    max_sequence_length = train_data.shape[1]
    vocabulary_size = len(reverse_word_index) +1
    embedding_dim = embedding_matrix.shape[1]
    category_number = 19
    input_shape = train_data.shape[1:]

    embedding_layer = Embedding(vocabulary_size,
                                embedding_dim,
                                weights = [embedding_matrix],
                                input_length=max_sequence_length,
                                trainable = False,
                                name = 'embedding_layer')
    model = LSTM_MODEL(input_shape,category_number,embedding_layer)


    if not os.path.isdir('../inputData/cache'):
        os.mkdir('../inputData/cache')

    weight_name = 'weights_LSTM.h5'
    weight_path = os.path.join("../inputData/cache", weight_name)

    if pre_train:
        model.load_weights(weight_path)


    print('checkpoint')
    checkpointer = ModelCheckpoint(filepath = weight_path, verbose = 1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose = 0, mode = 'auto')
    print('Early stop at 5')
    # fit the model
    History = model.fit(train_data, train_label,
              batch_size = batch_size,
              epochs = nb_epoch,
              validation_data = [val_data, val_label],
              callbacks = [checkpointer, earlystopping])

    # evaluate the model
    # summarize history for accuracy
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc = 'upper left')
    plt.show()

    # summerize history for loss
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    score = model.evaluate(test_data,test_label,verbose = 0)
    # mse - mean squared error; mae - mean absolute error; mape - mean absolute percentage error
    print('{}: {}'.format(model.metrics_names[1], score[1]*100))


print("---Load the raw data---")
df = pd.read_csv("../inputData/ICD9_Note.csv")

print("---Output the word_index file---")
data, word_index, reverse_word_index = createWordSequence(df,max_sequence_len = 50)
output_pickle(word_index, "../inputData/Data_WordIndex.p")

print("---Output the embedding matrix file---")
em = create_EmbeddingMatrix(word_index, "../inputData/glove.840B.300d.txt",remove_stopwords=True)
output_pickle(em,"../inputData/EmbMatrix_GloVe_300d.p")

print("---Create train-validate-test file---")
N = df.shape[0]
seed = 1000
idx_train, idx_val, idx_test = train_test_separator(seed,N)

train_data = data[idx_train, :]
train_label = df['ICD9_CODE'].values[idx_train]
print(train_label)
train_info = np.concatenate((train_label.reshape((-1,1)), train_data), axis = 1)
np.savetxt("../inputData/train.txt",train_info, fmt = '%i', delimiter=',')

val_data = data[idx_val, :]
val_label = df['ICD9_CODE'].values[idx_val]
validation_info = np.concatenate((val_label.reshape((-1,1)),val_data), axis = 1)
np.savetxt("../inputData/validation.txt",validation_info,fmt = '%i', delimiter=',')

test_data = data[idx_test,:]
test_label = df['ICD9_CODE'].values[idx_test]
test_info = np.concatenate((test_label.reshape((-1,1)), test_data), axis = 1)
np.savetxt("../inputData/test.txt",test_info,fmt = '%i', delimiter=',')

print("---Train model---")

# convert the label to one-hot vector
train_label_onehot = np.zeros((train_label.size,19))
train_label_onehot[np.arange(train_label.size),train_label] = 1

val_label_onehot = np.zeros((val_label.size, 19))
val_label_onehot[np.arange(val_label.size), val_label] = 1

test_label_onehot = np.zeros((test_label.size, 19))
test_label_onehot[np.arange(test_label.size), test_label] = 1
print("test_label_onehot")
print(test_label_onehot)

train(reverse_word_index, em, train_data, train_label_onehot, val_data, val_label_onehot, test_data, test_label_onehot)
print("---End---")