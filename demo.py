import con_util
reload(con_util)
from con_util import *
import os
import pickle
import sys

# Demo script for conspeech
# 
# Usage:
# python demo.py
# python demo.py [class]               Example: python demo.py RY
# python demo.py [class] [lambda]      Example: python demo.py RY 0.25

lambd = 0.5
speech_class = 'DN'

if len(sys.argv) >= 2:
    speech_class = sys.argv[1]
    if not speech_class in ['RY','RN','DN','DY']:
        print 'Invalid parameter:',speech_class
        sys.exit()
    
if len(sys.argv) >= 3:
    lambd = float(sys.argv[2])
    if (lambd < 0.0) or (lambd > 1.0):
        print 'Invalid parameter:',lambd
        sys.exit()
    

# Dataset from http://www.cs.cornell.edu/home/llee/data/convote.html
PATH_TO_DATA = 'convote_v1.1' + os.path.sep + 'data_stage_three'
TRAIN_DIR = os.path.join(PATH_TO_DATA, "training_set")
TEST_DIR = os.path.join(PATH_TO_DATA, "test_set")
DEV_DIR = os.path.join(PATH_TO_DATA, "development_set")

(dataset,vocab) = construct_dataset([TRAIN_DIR,TEST_DIR,DEV_DIR])

if sum([len(x) for x in dataset.values()]) == 0:
    print 'No data found!'
    sys.exit()
    
print '# Class: ' + speech_class + ', Lambda: ' + str(lambd) + ' #'


class_words = get_class_words(dataset)

jk = pickle.load( open( "jk.p", "rb" ) )
jk_trend = get_jk_trend(jk,print_n=0)

ngram_probs = get_n_gram_probs(dataset,n=6, verbose = False)
gen_sp =  generate_speech_wba(dataset,ngram_probs,None,None,jk_trend,jk,speech_class,lamb=lambd)
