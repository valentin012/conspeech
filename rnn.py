#!/usr/bin/env python3

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *
import os
import random
from collections import defaultdict
from scipy.stats import futil
import re
from rnn_theano import RNNTheano, gradient_check_theano
from utils import load_model_parameters_theano, save_model_parameters_theano


START_OF_SPEECH = "__START__"
END_OF_SPEECH = "__END__"
END_OF_SENTENCE = "__STOP__"
REFERENCE = "<ref>"
NUMBER = "<number>"

max_vocab_size = 6000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"



PATH_TO_DATA = 'convote_v1.1\data_stage_three'
TRAIN_DIR = os.path.join(PATH_TO_DATA, "training_set")
TEST_DIR = os.path.join(PATH_TO_DATA, "test_set")
DEV_DIR = os.path.join(PATH_TO_DATA, "development_set")

classes = ['DY','DN','RY','RN']



def construct_dataset(paths):
    print("[constructing dataset...]")

    class_sentences = dict()
    for c in classes:
        class_sentences[c] = []
        
    #for l in labels:
    #    dataset[l] = []

    for p in paths:
        for f in sorted(os.listdir(p)): 
            #006_400102_0002030_DON.txt
            vote = f[21:22]
            party = f[19:20]
            label = party + vote
            if label not in classes:
                continue;
            with open(os.path.join(p,f),'r') as doc:
                content = doc.read()

                content = content.replace('; center ', '; ')
                content = content.replace(' /center ', ' ')
                content = content.replace(' em ', ' ')
                content = content.replace(' /em ', ' ')
                content = content.replace(' pre ', ' ')
                content = content.replace(' /pre ', ' ')

                content = content.replace(' & lt ;', '')
                content = content.replace(' & gt ;', '')
                content = content.replace(' p ; ', ' ')
                content = content.replace(' & amp ; ', ' ')

                content = content.replace(' p nbsp ; ', ' ') 
                content = content.replace(' nbsp ;', '') 
                content = content.replace(' p ; ', ' ')
                content = content.replace(' p lt ;', '')
                content = content.replace(' p gt ;', '')

                content = content.replace(' b ', ' ')
                content = content.replace(' p ', ' ')

                content = content.replace(" n't", "n't")
                content = content.replace(" 's", "'s")   
                content = content.replace(" h. con .  res. ", " h.con.res. ")  
                content = content.replace('.these ', '. these ')

                content = re.sub(r'[a-z]\.[a-z] \.  ',lambda pat: pat.group(0).replace(' ','') + ' ',content)

                content = re.sub(r'xz[0-9]{7}',REFERENCE,content)
                #content = re.sub(r' [0-9]+ ', ' ' + NUMBER + ' ',content) 
                #content = re.sub(r' [0-9]+\.[0-9]+ ', ' ' + NUMBER + ' ',content) 

                #content = content.replace(' no .  ' + NUMBER, ' no. ' + NUMBER)
                content = re.sub(r' no .  [0-9]', lambda pat: pat.group(0).replace(' .  ','. ') + ' ',content)

                content = content.replace(chr(0xc3), '')
                content = content.replace(chr(0x90), '')

                #lines = content.split(" . ")
                lines = re.split(r' \.  | \!  | \?  ',content)
                lines = [x.strip() for x in lines]
                lines = [a for a in lines if a.strip() != '']

                if len(lines) <= 1:
                    continue


                for idx,line in enumerate(lines):
                    lines[idx] = sentence_start_token + ' ' + lines[idx] + ' ' + sentence_end_token


                #lines.insert(0,START_OF_SPEECH)
                #lines.append(END_OF_SPEECH)

                class_sentences[label].extend(lines)

    print("[dataset constructed.]")
    return class_sentences   



def generate_sentence(model,word_to_index,index_to_word):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

   

    
if __name__=='__main__':   
    
    # Download NLTK model data (you need to do this once)
    nltk.download("book")
    
    dataset = construct_dataset([TRAIN_DIR,TEST_DIR,DEV_DIR])
    print("Sentences", sum([len(x) for x in dataset.values()]))
    
    for label, sentences in dataset.items():
        print('Processing', label, '...')

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))
        
        vocabulary_size = min(max_vocab_size, len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        #print "\nExample sentence: '%s'" % sentences[0]
        #print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
                              
        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
                              
        model = RNNTheano(vocabulary_size, hidden_dim=50)
        losses = train_with_sgd(model, X_train, y_train, nepoch=50)
        save_model_parameters_theano('./data/trained-model-'+label+'-dim50-t50.npz', model)
        #load_model_parameters_theano('./data/trained-model-theano.npz', model)
        
                              
