"""
Main trainer function
"""
import theano
import theano.tensor as tensor
#import Pickle as pkl
import numpy
import copy
import os
import tensorflow as tf
from collections import Counter
import _pickle as pkl
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import collections
import sys
from skip_thoughts.data import special_words
import time
#import pickle
import homogeneous_data
#from special_words import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from utils import*
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model
from vocab import load_dictionary
input_directory='C:/Users/Hp 15/Desktop/pos/'
input_files1=glob.glob(input_directory+"*.txt")
for path in input_files1:
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()    
        keys_list = [keys for keys in flags_dict]    
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    del_all_flags(tf.flags.FLAGS)
    # because path is object not string
    input_files= str(path)
FLAGS = tf.flags.FLAGS
word_idict="C:/Users/Hp 15/Desktop/vocab/vocab.txt"
tf.flags.DEFINE_string("vocab_file", word_idict,
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")

def _build_vocabulary(input_files):

    if FLAGS.vocab_file:
    
       tf.logging.info("Loading existing word_idict file.")
       word_idict = collections.OrderedDict()
       with tf.gfile.GFile(FLAGS.vocab_file, mode="r") as f:
           for i, line in enumerate(f):
                word = line.strip()
                assert word not in word_idict, "Attempting to add word twice: %s" % word
                word_idict[word] = i
           tf.logging.info("Read word_idict of size %d from %s",
                    len(word_idict), FLAGS.vocab_file)
           return word_idict



def trainer(X, 
            dim_word=620, # word vector dimensionality
            dim=2400, # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000,
            maxlen_w=30,
            optimizer='adam',
            batch_size = 64,
            saveto='C:/Users/Hp 15/Desktop/vocab/',
            
            dictionary='C:/Users/Hp 15/Desktop/vocab/vocab.txt',
            saveFreq=1000,
            reload_=False):
    print ("Done heloooooooshabnam")
    # Model options
    model_options = {}
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['decoder'] = decoder 
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['saveFreq'] = saveFreq
    model_options['reload_'] = reload_

    print (model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print ("reloading...") + saveto
        

        file = open('C:/Users/Hp 15/Desktop/vocab/vocab.txt', "r") 
        models_options=file.read()
#        with open('C:/Users/Hp 15/Desktop/vocab/vocab.txt', 'r') as f:
#            models_options =pkl.load(f)
    file = open('C:/Users/Hp 15/Desktop/vocab/word_counts.txt', "r") 
    wordcount=file.read()
    words = list(wordcount)
    freqs = Counter(file.read().split())
#    freqs = list(wordcount.values())
    sorted_indices = np.argsort(freqs)[::-1] 
#have to look again    # load dictionary
    print ("Loading dictionary...")
    worddict = load_dictionary(dictionary)

    # Inverse dictionary
    word_idict = dict()
    word_idict = collections.OrderedDict()
    word_idict[special_words.EOS] = special_words.EOS_ID
    word_idict[special_words.UNK] = special_words.UNK_ID
    for kk , vv in enumerate(sorted_indices[0:-2]):
       word_idict[words[vv]] = kk + 2  # 0: EOS, 1: UNK.
    
    
    
    
#    
#    for kk, vv in worddict.iteritems():
#        word_idict[vv] = kk
#    word_idict[0] = '<eos>'
#    word_idict[1] = 'UNK'

    print ("Building model")
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, x, x_mask, y, y_mask, z, z_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, z, z_mask]

    # before any regularizer
    print ("Building f_log_probs..."),
    f_log_probs = theano.function(inps, cost, profile=False)
    print ("Done")

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print ("Building f_cost..."),
    f_cost = theano.function(inps, cost, profile=False)
    print ("Done")

    print ("Done")
    print ("Building f_grad..."),
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print ("Building optimizers..."),
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print ("Optimization")

    # Each sentence in the minibatch have same length (for encoder)
    trainX = homogeneous_data.grouper(X)
    train_iter = homogeneous_data.HomogeneousData(trainX, batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    lrate = 0.01
    for eidx in xrange(max_epochs):
        n_samples = 0

        print ("Epoch "), eidx

        for x, y, z in train_iter:
            n_samples += len(x)
            uidx += 1

            x, x_mask, y, y_mask, z, z_mask = homogeneous_data.prepare_data(x, y, z, worddict, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print ("Minibatch with zero sample under length "), maxlen_w
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, x_mask, y, y_mask, z, z_mask)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print ("NaN detected")
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print ("Epoch"), eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, saveFreq) == 0:
                print ("Saving..."),

                params = unzip(tparams)
                numpy.savez(saveto, history_errs=[], **params)
                pkl.dump(model_options, open('C:/Users/Hp 15/Desktop/vocab/', 'wb'))
                print ("Done")

        print ("Seen %d samples")+n_samples

if __name__ == '__main__':
    print ("Done shabnam")
    file = open('C:/Users/Hp 15/Desktop/11525.txt', "r") 
    X=file.read()
    trainer(X, 
            dim_word=620, # word vector dimensionality
            dim=2400, # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000,
            maxlen_w=30,
            optimizer='adam',
            batch_size = 64,
            saveto='C:/Users/Hp 15/Desktop/vocab/',
            
            dictionary='C:/Users/Hp 15/Desktop/vocab/vocab.txt',
            saveFreq=1000,
            reload_=False)
    print (".npz shabnam")
    pass



