"""
Constructing and loading dictionaries
"""
import glob
import _pickle as pkl
import numpy
from collections import OrderedDict
text=glob.glob('C:/Users/Hp 15/Desktop/desktop/new/'+"*.txt")
#text=['hellooooooooooo', 'tooooooooooooooooooogggggg', 'okoko','ijiji']
def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())[0]
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>

    return worddict, wordcount

def load_dictionary(loc='C:/Users/Hp 15/Desktop/vocab/vocab.txt'):
    """
    Load a dictionary
    """
    print ("Done fffffffffffffffffffff")
    file = open('C:/Users/Hp 15/Desktop/vocab/vocab.txt', "r") 
    worddict =file.read()
#    with open(loc, 'rb') as f:
#        worddict = pkl.load(f)
    return worddict

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

if __name__ == "__main__":
    X=glob.glob('C:/Users/Hp 15/Desktop/desktop/new/'+"*.txt")
    worddict, wordcount = build_dictionary(X)
    
    save_dictionary(worddict, wordcount, loc='C:/Users/Hp 15/Desktop/desktop/opcodes/22.txt')
    
    load_dictionary(loc='C:/Users/Hp 15/Desktop/desktop/opcodes/22.txt')
