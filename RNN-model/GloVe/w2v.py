import numpy as np
import time
import collections
import tensorflow as tf
import random

from tensorflow.contrib import rnn
from scipy import spatial
from nltk.corpus import stopwords

start = time.time()

file = "../inputData/glove.840B.300d.txt"
fable_text = """
long ago , the mice had a general council to consider what measures
they could take to outwit their common enemy , the cat . some said
this , and some said that but at last a young mouse got up and said
he had a proposal to make , which he thought would meet the case . 
you will all agree , said he , that our chief danger consists in the
sly and treacherous manner in which the enemy approaches us . now , 
if we could receive some signal of her approach , we could easily
escape from her . i venture , therefore , to propose that a small
bell be procured , and attached by a ribbon round the neck of the cat
. by this means we should always know when she was about , and could
easily retire while she was in the neighbourhood . this proposal met
with general applause , until an old mouse got up and said that is
all very well , but who is to bell the cat ? the mice looked at one
another and nobody spoke . then the old mouse said it is easy to
propose impossible remedies .
"""

fable_text = fable_text.replace("\n", " ") # puts all the words in a single row

# Load the pre-trained GloVe
def loadGloveModel(gloveFile):
    print("Loading GloVe Model")

    with open(gloveFile) as f:
        content = f.readlines()
    model = {}

    for line in content:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    embedding_dim = len(embedding)
    print("Done.", len(model), "words loaded!")
    print("It took", time.time() - start, 'seconds.')
    return model, embedding_dim

# Load the text file and puts all the words in a single column vector within a numpy array
def read_data(raw_text):
    whitelist = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    content = ''.join(filter(whitelist.__contains__, raw_text))
    content = content.split() # splits the text by spaces
    content = np.array(content)
    content = np.reshape(content,[-1, ])
    return content

# Create dictionary and reverse dictionary with word ids
def build_dictionaries(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


model, embedding_dim = loadGloveModel(file)
training_data = read_data(fable_text)
dictionary, reverse_dictionary = build_dictionaries(training_data)

# Create embedding array
doc_vocab_size = len(dictionary)
dict_as_list = sorted(dictionary.items(), key = lambda x:x[1])
embeddings_temp = []
for i in range(doc_vocab_size):
    item = dict_as_list[i][0]
    if item in model:
        embeddings_temp.append(model[item])
    else:
        rand_num = np.random.uniform(low = 0.2, high = 0.2, size = embedding_dim)
        embeddings_temp.append(rand_num)

# final embedding array corresponding to dictionary of words in the document
embedding = np.array(embeddings_temp)

# create tree so that we can later search for closest vector to predcition
tree = spatial.KDTree(embedding)

# model parameters
learning_rate = 0.001
n_input = 3 # this is the number of words that are read at a time
n_hidden = 512

# create input placeholders
x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.float32, [None, embedding_dim])

# RNN output node weights and biases
weights = {'out': tf.Variable(tf.random_normal([n_hidden, embedding_dim]))}
biases = {'out': tf.Variable(tf.random_normal([embedding_dim]))}


with tf.name_scope("embedding"):
    W = tf.Variable(tf.constant(0.0, shape = [doc_vocab_size, embedding_dim]), trainable = True, name = "W")
    embedding_placeholder = tf.placeholder(tf.float32, [doc_vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    embedded_chars = tf.nn.embedding_lookup(W,x)

# reshape input data
x_unstack = tf.unstack(embedded_chars, n_input, 1)

# create LSTM cells
rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
outputs, states = rnn.static_rnn(rnn_cell, x_unstack, dtype = tf.float32)

# capture only the last output
pred = tf.matmul(outputs[-1], weights['out']+biases['out'])

# create the loss function and optimizer with clipped gradient
cost = tf. reduce_mean(tf.nn.l2_loss(pred-y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(cost)
clipped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(clipped_gvs)

# Initilaize
init = tf.global_variables_initializer()

# lanch the graph
sess = tf.Session()
sess.run(init)
sess.run(embedding_init, feed_dict = {embedding_placeholder: embedding})

step = 0
offset = random.randint(0, n_input + 1) # random integer
end_offset = n_input + 1
acc_total = 0
loss_total = 0
training_iters = 10000
display_step = 100

while step < training_iters:
    ### Generate a minibatch ###
    # when offset gets close to the end of training data, restart near the beginning
    if offset > (len(training_data) - end_offset):
        offset = random.randint(0, n_input+1)

    # get the integer representation for the input words
    x_integers = [[dictionary[str(training_data[i])]] for i in range(offset, offset+ n_input)]
    x_integers = np.reshape(np.array(x_integers), [-1, n_input])

    # create embedding for target vector
    y_position = offset + n_input
    y_integer = dictionary[training_data[y_position]]
    y_embedding = embedding[y_integer, :]
    y_embedding = np.reshape(y_embedding, [1,-1])

    _,loss,pred_ = sess.run([train_op, cost, pred], feed_dict= {x: x_integers, y: y_embedding})
    loss_total += loss

    # display output to show progress

    if(step +1)% display_step == 0:
        words_in = [str(training_data[i]) for i in range(offset, offset + n_input)]
        target_word = str(training_data[y_position])

        nearest_dist, nearest_idx = tree.query(pred_[0],3)
        nearest_words = [reverse_dictionary[idx] for idx in nearest_idx]

        print("{} - {} vs {}".format(words_in, target_word, nearest_words))
        print("Average Loss = {0:.6g}".format(loss_total/display_step))
        loss_total = 0

    step += 1
    offset += (n_input + 1)
print("Finished Optimization. step ", step)
