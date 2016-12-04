######################################################################
#  CliNER - keras_ml.py                                              #
#                                                                    #
#  Willie Boag                                                       #
#                                                                    #
#  Purpose: An interface to the Keras library.                       #
######################################################################

__author__ = 'Willie Boag'
__date__   = 'Aug. 18, 2016'

import numpy as np
import os
import random
import time

import tensorflow as tf
tf.python.control_flow_ops = tf
tf.python.ops = tf

from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Masking

from keras import backend as K
K.set_learning_phase(0)

# only load compile this model once per run (useful when predicting many times)
lstm_model = None
max_char_index = 50
max_word_size = 25



def train(train_word_X_ids, train_char_X_ids, train_Y_ids, tag2id,
          W=None, epochs=10, val_X_ids=None, val_Y_ids=None):
    '''
    train()

    Build a Keras Bi-LSTM and return an encoding of it's parameters for predicting.

    @param train_X_ids.  A list of tokenized sents (each sent is a list of num ids)
    @param train_Y_ids.  A list of concept labels parallel to train_X_ids
    @param W.            Optional initialized word embedding matrix.
    @param epochs.       Optional number of epochs to train model for.
    @param val_X_ids.    A list of tokenized sents (each sent is a list of num ids)
    @param val_Y_ids.    A list of concept labels parallel to train_X_ids

    @return A tuple of encoded parameter weights and hyperparameters for predicting.
    '''
    # gotta beef it up sometimes
    # (I know this supposed to be the same as 5x more epochs,
    #    but it doesnt feel like it)
    #train_X_ids = train_X_ids * 15
    #train_Y_ids = train_Y_ids * 15

    # build model
    char_input_dim    = 83
    """for i, sent in enumerate(train_char_X_ids):
        max_char_id_in_sent = max(map(max, sent))
        if(char_input_dim < max_char_id_in_sent):
            char_input_dim = max_char_id_in_sent"""

    char_maxlen        = 50
    """for i, sent in enumerate(train_char_X_ids):
        max_wordlen_in_sent = max(map(len, train_char_X_ids[i]))
        if(char_maxlen < max_wordlen_in_sent):
             char_maxlen = max_wordlen_in_sent
    print char_maxlen, "please record this"
    """


    word_input_dim    = max(map(max, train_word_X_ids)) + 1
    word_maxlen       = max(map(len, train_word_X_ids))
    num_tags     = len(tag2id)

    lstm_model = create_bidirectional_lstm(word_input_dim=word_input_dim, char_input_dim=char_input_dim, word_maxlen=word_maxlen, char_maxlen=char_maxlen, nb_classes=num_tags, W=W)

    # turn each id in Y_ids into a onehot vector
    train_Y_seq_onehots = [to_categorical(y, nb_classes=num_tags) for y in train_Y_ids]

    # format X and Y data
    nb_samples = len(train_word_X_ids)
    train_X_words = create_data_matrix_X(train_word_X_ids, nb_samples, word_maxlen, num_tags)
    train_X_chars = create_data_matrix_X_chars(train_char_X_ids, nb_samples, word_maxlen, char_maxlen)
    train_Y = create_data_matrix_Y(train_Y_seq_onehots, nb_samples, word_maxlen, num_tags)

    # fit model
    print 'training begin'
    batch_size = 64
    #'''
    history = lstm_model.fit([train_X_chars, train_X_words], train_Y,
                             batch_size=batch_size, nb_epoch=epochs, verbose=1)
    #'''
    #history = {}
    print 'training done'

    ######################################################################

    # information about fitting the model
    hyperparams = batch_size, num_tags, word_maxlen
    scores = {}
    scores['train'] = compute_stats('train', lstm_model, hyperparams,
                                    train_X_words, train_X_chars, train_Y_ids)
    #if val_X_ids:
    #    val_X = create_data_matrix_X(val_X_ids, len(val_X_ids), word_maxlen, num_tags)
    #    scores['dev'] = compute_stats('dev', lstm_model, hyperparams,
    #                                  val_X, val_Y_ids)
    scores['history'] = history.history

    ######################################################################

    # needs to return something pickle-able
    param_filename = '/tmp/tmp_keras_weights-%d' % random.randint(0,9999)
    lstm_model.save_weights(param_filename)
    with open(param_filename, 'rb') as f:
        lstm_model_str = f.read()
    os.remove(param_filename)

    # return model back to cliner
    keras_model_tuple = (lstm_model_str, word_input_dim, char_input_dim, num_tags, word_maxlen, char_maxlen)

    return keras_model_tuple, scores




def predict(keras_model_tuple, X_seq_ids, X_char_ids):
    '''
    predict()

    Predict concept labels for X_seq_ids using Keras Bi-LSTM.

    @param keras_model_tuple.  A tuple of encoded parameter weights and hyperparams.
    @param X_seq_ids.          A list of tokenized sents (each is a list of num ids)

    @return  A list of concept labels parallel to train_X_ids
    '''
    global lstm_model

    # unpack model metadata
    lstm_model_str, word_input_dim, char_input_dim, num_tags, word_maxlen, char_maxlen = keras_model_tuple

    # build LSTM once (weird errors if re-compiled many times)
    if lstm_model is None:
        lstm_model = create_bidirectional_lstm(word_input_dim=word_input_dim, char_input_dim=char_input_dim, word_maxlen=word_maxlen, char_maxlen=char_maxlen, nb_classes=num_tags)

    # dump serialized model out to file in order to load it
    param_filename = '/tmp/tmp_keras_weights-%d' % random.randint(0,9999)
    with open(param_filename, 'wb') as f:
        f.write(lstm_model_str)

    # load weights from serialized file
    lstm_model.load_weights(param_filename)
    os.remove(param_filename)

    # format data for LSTM
    nb_samples = len(X_seq_ids)
    X = create_data_matrix_X(X_seq_ids, nb_samples, word_maxlen, num_tags)
    X_chars = create_data_matrix_X_chars(X_char_ids, nb_samples, word_maxlen, char_maxlen)

    # Predict tags using LSTM
    batch_size = 128
    p = lstm_model.predict([X_chars, X], batch_size=batch_size)
    # Greedy decoding of predictions
    # TODO - this could actually be the perfect spot for correcting O-before-I tags
    predictions = []
    for i in range(nb_samples):
        num_words = len(X_seq_ids[i])
        if num_words <= word_maxlen:
            tags = p[i,word_maxlen-num_words:].argmax(axis=1)
            predictions.append(tags.tolist())
        else:
            # if the sentence had more words than the longest sentence
            #   in the training set
            residual_zeros = [ 0 for _ in range(num_words-word_maxlen) ]
            padded = list(p[i].argmax(axis=1)) + residual_zeros
            predictions.append(padded)
    print predictions

    return predictions



def compute_stats(label, lstm_model, hyperparams, X_words, X_chars, Y_ids):
    '''
    compute_stats()

    Compute the P, R, and F for a given model on some data.

    @param label.        A name for the data (e.g. "train" or "dev")
    @param lstm_model.   The trained Keras model
    @param hyperparams.  A tuple of values for things like num_tags and batch_size
    @param X.            A formatted collection of input examples
    @param Y_ids.        A list of list of tags - the labels to X.
    '''
    # un-pack hyperparameters
    batch_size, num_tags, maxlen = hyperparams


    # predict label probabilities
    pred = lstm_model.predict([X_chars, X_words], batch_size=batch_size)

    # choose the highest-probability labels
    nb_samples = len(Y_ids)
    predictions = []
    for i in range(nb_samples):
        num_words = len(Y_ids[i])
        tags = pred[i,maxlen-num_words:].argmax(axis=1)
        predictions.append(tags.tolist())

    # confusion matrix
    confusion = np.zeros( (num_tags,num_tags) )
    for tags,yseq in zip(predictions,Y_ids):
        for y,p in zip(yseq, tags):
            confusion[p,y] += 1

    # print confusion matrix
    print '\n'
    print label
    print ' '*6,
    for i in range(num_tags):
        print '%4d' % i,
    print ' (gold)'
    for i in range(num_tags):
        print '%2d' % i, '   ',
        for j in range(num_tags):
            print '%4d' % confusion[i][j],
        print
    print '(pred)'
    print '\n'

    precision = np.zeros(num_tags)
    recall    = np.zeros(num_tags)
    f1        = np.zeros(num_tags)

    for i in range(num_tags):
        correct    =     confusion[i,i]
        num_pred   = sum(confusion[i,:])
        num_actual = sum(confusion[:,i])

        p  = correct / (num_pred   + 1e-9)
        r  = correct / (num_actual + 1e-9)

        precision[i] = p
        recall[i]    = r
        f1[i]        = (2*p*r) / (p + r + 1e-9)

    scores = {}
    scores['precision'] = precision
    scores['recall'   ] = recall
    scores['f1'       ] = f1

    return scores

def create_bidirectional_lstm(word_input_dim, char_input_dim, word_maxlen, char_maxlen, nb_classes, W=None):
    # model will expect: (nb_samples, timesteps, input_dim)

    char_input = Input(shape=(char_maxlen,), dtype='int32')

    char_embedding = Embedding(output_dim=50, input_dim=char_input_dim, input_length=char_maxlen, mask_zero=True)(char_input)

    char_LSTM_f = LSTM(output_dim=50)(char_embedding)
    char_LSTM_r = LSTM(output_dim=50, go_backwards=True)(char_embedding)
    char_LSTM_fr = merge([char_LSTM_f, char_LSTM_r], mode='concat', concat_axis=-1)
    char_encoder_fr = Model(input=char_input, output=char_LSTM_fr)

    # apply char level encoder to every character sequence
    char_seqs = Input(shape=(word_maxlen, char_maxlen), dtype='int32', name='char')
    encoded_char_fr_states = TimeDistributed(char_encoder_fr)(char_seqs)
    # m_encoded_char_fr_states = Masking(0.0)(encoded_char_fr_states)
    m_encoded_char_fr_states = (encoded_char_fr_states)


    # input tensor
    word_input = Input(shape=(word_maxlen,), dtype='int32')

    # initialize Embedding layer with pretrained vectors
    if W is not None:
        embedding_size = W.shape[1]
        weights = [W]
    else:
        embedding_size = 300
        weights = None

    # Embedding layers
    embedding_word = Embedding(output_dim=embedding_size, input_dim=word_input_dim, input_length=word_maxlen, mask_zero=False, weights=weights)(word_input)
    merged_embeddings = merge([embedding_word, m_encoded_char_fr_states], mode='concat', concat_axis=-1)

    # LSTM 1 input
    hidden_units = 128
    lstm_f1 = LSTM(output_dim=hidden_units,return_sequences=True)(embedding_word)
    lstm_r1 = LSTM(output_dim=hidden_units,return_sequences=True,go_backwards=True)(embedding_word)
    merged1 = merge([lstm_f1, lstm_r1], mode='concat', concat_axis=-1)

    # LSTM 2 input
    lstm_f2 = LSTM(output_dim=hidden_units,return_sequences=True)(merged1)
    lstm_r2 = LSTM(output_dim=hidden_units,return_sequences=True,go_backwards=True)(merged1)
    merged2 = merge([lstm_f2, lstm_r2], mode='concat', concat_axis=-1)

    # Dropout
    after_dp = TimeDistributed(Dropout(0.5))(merged2)

    # fully connected layer
    fc1 = TimeDistributed(Dense(output_dim=128, activation='sigmoid'))(after_dp)
    fc2 = TimeDistributed(Dense(output_dim=nb_classes, activation='softmax'))(fc1)

    model = Model(input=[char_seqs, word_input], output=fc2)

    print
    print 'compiling model'
    start = time.clock()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    end = time.clock()
    print 'finished compiling: ', (end-start)
    print

    return model


def create_data_matrix_X_chars(X_char_ids, nb_samples, word_maxlen, char_maxlen):
    X = np.zeros(shape=(nb_samples, word_maxlen, char_maxlen))

    for i in range(nb_samples):
        # length of cur sentence
        cur_sent_len = len(X_char_ids[i])
        # length of words in cur sentence
        cur_word_lens = map(len, X_char_ids[i])

        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if word_maxlen - cur_sent_len < 0:
            cur_len = word_maxlen

        for j, word_len in enumerate(cur_word_lens):
            if char_maxlen < word_len:
                cur_word_lens[j] = char_maxlen

        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        for k in range(len(X_char_ids[i])):
            if k < word_maxlen:
                X[i, word_maxlen - len(X_char_ids[i]) + k, char_maxlen - cur_word_lens[k]:] = X_char_ids[i][k][:cur_word_lens[k]]
    return X

def create_data_matrix_X(X_ids, nb_samples, maxlen, nb_classes):
    X = np.zeros((nb_samples, maxlen))

    for i in range(nb_samples):
        cur_len = len(X_ids[i])

        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if maxlen-cur_len < 0:
            cur_len = maxlen

        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        X[i, maxlen - cur_len:] = X_ids[i][:maxlen]

    return X



def create_data_matrix_Y(Y_seq_onehots, nb_samples, maxlen, nb_classes):
    Y = np.zeros((nb_samples, maxlen, nb_classes))

    for i in range(nb_samples):
        cur_len = len(Y_seq_onehots[i])

        # ignore tail of sentences longer than what was trained on
        #    (only happens during prediction)
        if maxlen-cur_len < 0:
            cur_len = maxlen

        # We pad on the left with zeros,
        #    so for short sentences the first elemnts in the matrix are zeros
        Y[i, maxlen - cur_len:, :] = Y_seq_onehots[i][:maxlen]

    return Y
