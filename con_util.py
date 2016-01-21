from __future__ import division
import operator
import os
import random
from collections import defaultdict
from scipy.stats import futil
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import re
import sys
from nltk import pos_tag


# Project: Political Speech Generator
# Author:  Valentin Kassarnig
# Email:   valentin.kassarnig@gmail.com


START_OF_SPEECH = "__START__"
END_OF_SPEECH = "__END__"
END_OF_SENTENCE = "__STOP__"
REFERENCE = "<ref>"
NUMBER = "<number>"



classes = ['DY','DN','RY','RN']


vocab_count = defaultdict(float)

def construct_dataset(paths):
    print "[constructing dataset...]"
    dataset = dict()
    
    for c in classes:
        dataset[c] = []
        
    vocab = set()
    vocab.add(START_OF_SPEECH)
    vocab.add(END_OF_SPEECH)
    vocab.add(END_OF_SENTENCE)
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
                lines = filter(lambda a: (a.strip() != ''), lines)

                if len(lines) <= 1:
                    continue


                for idx,line in enumerate(lines):
                    lines[idx] = lines[idx] + ' ' + END_OF_SENTENCE

                    words = line.split();
                    for word in words:
                        vocab.add(word)
                        vocab_count[word] += 1

                lines.insert(0,START_OF_SPEECH)
                lines.append(END_OF_SPEECH)

                dataset[label].append(lines)

    print "[dataset constructed.]"
    return (dataset,vocab)               


def get_class_words(dataset):
    class_words = dict()    

    for c in classes:
        class_words[c] = defaultdict(float)
    
    for key,speeches in dataset.iteritems():
        for speech in speeches:
            for sentence in speech:
                for word in sentence.split():
                    class_words[key][word] += 1
    return class_words


def jk_pos_tag_filter(dataset):
    #Justeson and Katz Filter
    import nltk
    from nltk import pos_tag
    import sys
    import pickle

    jk_trigram_filter_ = [['NN','NN','NN'],['JJ','JJ','NN'],['JJ','NN','NN'],['NN','JJ','NN'],['NN','IN','NN'],['NN','CC','NN']]
    jk_bigram_filter = [['NN','NN'],['JJ','NN']]
    #nltk.download('maxent_treebank_pos_tagger');

    jk = dict()
    for c in classes:
        jk[c] =defaultdict(float)

    speech_cnt = 0
    for key,speeches in dataset.iteritems():
        print key
        sys.stdout.flush()
        for idx,speech in enumerate(speeches):
            for sentence in speech:
                words = sentence.split()
                if len(words) < 3:
                    continue

                tags = pos_tag(words)            
                if ([tags[0][1], tags[1][1]] in jk_bigram_filter) and (tags[2][1] is not 'NN'):
                    tw = tags[0][0]+' '+tags[1][0]
                    jk[key][tw]+=1

                for i in range(len(tags)-2):
                    t = [tags[i][1], tags[i+1][1] ,tags[i+2][1]]
                    if t in jk_trigram_filter_:
                        tw = tags[i][0]+' '+tags[i+1][0]+' '+tags[i+2][0]
                        jk[key][tw]+=1
                    else:
                        t = [tags[i+1][1], tags[i+2][1]]
                        if t in jk_bigram_filter:
                            tw = tags[i+1][0]+' '+tags[i+2][0]
                            jk[key][tw]+=1

            if idx % 100 == 0:
                print idx,'/',len(speeches),'...'
                sys.stdout.flush()           

    
    return jk
    
    
def get_jk_trend(jk,print_n=10,thresh=1.0,min_occ=20):
    jk_trend = dict()
    totsum = 0

    for c in classes:
        jk_trend[c] =defaultdict(float)
        totsum += sum(jk[c].values())

    for c in classes:
        sorted_jk = sorted(jk[c].items(), key=operator.itemgetter(1),reverse=True)
        class_sum = sum(jk[c].values())
        for f in sorted_jk:
            #if f[1] < 2:
            #    continue

            p = f[1]/class_sum

            other_p = 0
            for c2 in classes:
                other_p += jk[c2][f[0]]
            other_p = other_p / totsum

            jk_trend[c][f[0]] = p/other_p

    for c in classes:
        if print_n > 0:
            print c 
            
        remlist = []
        for word, ratio in jk_trend[c].iteritems():
            if (ratio > thresh) and (sum([jk[x][word] for x in classes]) >= min_occ):
                pass
            else:
                remlist.append(word)
        for r in remlist:
            del jk_trend[c][r]
        sorted_jk = sorted(jk_trend[c].items(), key=operator.itemgetter(1),reverse=True)
        for sj in sorted_jk[:print_n]:
            print sj[0]
        #print len(jk_trend[c])
    return jk_trend

def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return len(s1[x_longest - longest: x_longest])
        
def generate_speech_sba(label,dataset,jk_trend,rand_set_size=20,sim_thresh = 0.1,max_sentences=30):    
    from nltk import trigrams
    
    print label
    random.seed()
    last_speech = dataset[label][random.randint(0,len(dataset[label])-1)]
    last_idx = 1
    last_sentence = last_speech[last_idx]

    speech_cnt = 0

    max_struc_sim = 0
    max_text_sim = 0

    

    print last_sentence
    sys.stdout.flush()
    for i in range(max_sentences):
        D=[]
        random.seed()
        while len(D) < rand_set_size:
            idx = random.randint(0,len(dataset[label])-1)
            sp = dataset[label][idx]
            if sp != last_speech:
                D.append(sp)

        max_similarity = 0.0
        max_struc_sim = 0
        max_text_sim = 0

        last_topics = []
        full_speech = " ".join(last_speech)
        for topic in jk_trend[label].keys():
            if topic in full_speech:
                last_topics.append(topic)
        """
        if (last_idx-1)/(len(last_speech)-2) <= 1/3:
            last_part = 1
        elif (last_idx-1)/(len(last_speech)-2) >= 2/3:
            last_part = 3
        else:
            last_part = 2
        #print last_part
        """

        last_tags = [x[1] for x in pos_tag(last_sentence.split())]
        last_tg = list(trigrams(last_sentence.split()))

        for speech in D:
            topic_cnt = 0
            full_speech = " ".join(speech)
            for topic in last_topics:
                if topic in full_speech:
                    topic_cnt += 1

            for idx,sentence in enumerate(speech):
                #print sentence
                similarity = 0.0
                struc_sim = 0
                text_sim = 0

                if (last_idx != 1) and (idx <= 1):
                    continue

                if (len(sentence.split()) <= 1):
                    continue


                tg = list(trigrams(sentence.split()))

                #for tg1 in last_tg:                
                #    for tg2 in tg:
                #        if tg1 == tg2:
                #            text_sim += 1
                #            break
                text_sim = len(set(last_tg) & set(tg))            
                text_sim = text_sim/(min(len(set(last_tg)),len(set(tg)))+0.01)            


                tags = [x[1] for x in pos_tag(sentence.split())]
                struc_sim = (longest_common_substring(last_tags,tags)) / (max(len(last_tags),len(tags)))           
                similarity = ((struc_sim)+(text_sim*3))
                #similarity = (text_sim)

                """
                if (idx-1)/(len(speech)-2) <= 1/3:
                    part = 1
                elif (idx-1)/(len(speech)-2) >= 2/3:
                    part = 3
                else:
                    part = 2

                if (last_part == 1) and (part == 3):
                    continue                
                if (part < last_part):
                    continue

                #if part == last_part:
                #    similarity += similarity

                #Same topics
                for i in range(topic_cnt):
                    similarity += similarity  
                """
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_struc_sim = struc_sim
                    max_text_sim = text_sim

                    if similarity > sim_thresh:
                        last_speech = speech
                        last_idx = idx+1 


        if max_similarity <= sim_thresh:
            last_idx += 1
        else:
            speech_cnt += 1

        last_sentence = last_speech[last_idx]

        #print last_speech[last_idx-1]
        #print 'Similarity:',max_similarity,'/ Struc:',max_struc_sim,'/ Text:',max_text_sim
        print last_sentence
        sys.stdout.flush()
        if last_sentence == END_OF_SPEECH:
            break
            
    print speech_cnt
    
    
def get_n_gram_class_probs(dataset,n=6):
    from nltk.util import ngrams

    class_tokens = dict()
    for c in classes:
        class_tokens[c] = []


    for key,speeches in dataset.iteritems():
        for speech in speeches:
            for sentence in speech:
                class_tokens[key].extend(sentence.split())

    #print len(tokens)
    n_gram_count = dict()
    n_gram_class_probs = dict()
    for c,tokens in class_tokens.iteritems():
        n_grams = ngrams(tokens,n)
        
        for ng in n_grams:
            if (END_OF_SPEECH in ng[:-1]):
                continue
                
            if ng not in n_gram_count:
                n_gram_count[ng] = defaultdict(float)
                n_gram_class_probs[ng] = defaultdict(float)
                
            n_gram_count[ng][c] += 1
            
    for n_gram,class_counts in n_gram_count.iteritems():
        for c in classes:
            n_gram_class_probs[n_gram][c] = class_counts[c]/sum(class_counts.values())

        
    return n_gram_class_probs
        
def get_n_gram_probs(dataset,n=6,verbose=True):
    from nltk.util import ngrams
    from nltk import trigrams
    from nltk import bigrams

    class_tokens = dict()
    for c in classes:
        class_tokens[c] = []


    for key,speeches in dataset.iteritems():
        for speech in speeches:
            for sentence in speech:
                class_tokens[key].extend(sentence.split())

    #print len(tokens)
    class_n_gram_probs = dict()
    for c,tokens in class_tokens.iteritems():
        n_grams = ngrams(tokens,n)


        n_gram_count = defaultdict(float)
        for ng in n_grams:
            if (END_OF_SPEECH in ng[:-1]):
                continue
            n_gram_count[ng] += 1

        prob = dict()
        for key, value in n_gram_count.iteritems():
            n_1_gram = tuple(key[:-1])
            word = key[-1]
            if n_1_gram not in prob:
                prob[n_1_gram] = defaultdict(float)
            prob[n_1_gram][word] += value

        for n_1_gram, words in prob.iteritems():
            n_1_gram_sum = sum(words.values())
            for word,cnt in words.iteritems():
                prob[n_1_gram][word] = prob[n_1_gram][word]/n_1_gram_sum

        for key, value in prob.iteritems():
            prob[key] = sorted(value.items(), key=operator.itemgetter(1), reverse=True)

        #n_gram_probs = sorted(prob.items(), key=lambda x: len(x[1]), reverse= True)
        class_n_gram_probs[c] = prob
        if verbose == True:
            print c,len(prob)
    return class_n_gram_probs


def get_corpus_n_gram_probs(dataset,n=6):
    from nltk.util import ngrams
    
    
    
    all_tokens = []

    for key,speeches in dataset.iteritems():
        for speech in speeches:
            for sentence in speech:
                all_tokens.extend(sentence.split())


    n_grams = ngrams(all_tokens,n)
    
    n_gram_count = defaultdict(float)
    
    for ng in n_grams:
        if (END_OF_SPEECH in ng[:-1]):
            continue
        n_gram_count[ng] += 1

    n_gram_probs = dict()
    for key, value in n_gram_count.iteritems():
        n_1_gram = tuple(key[:-1])
        word = key[-1]
        if n_1_gram not in n_gram_probs:
            n_gram_probs[n_1_gram] = defaultdict(float)
        n_gram_probs[n_1_gram][word] += value

    for n_1_gram, words in n_gram_probs.iteritems():
        n_1_gram_sum = sum(words.values())
        for word,cnt in words.iteritems():
            n_gram_probs[n_1_gram][word] = n_gram_probs[n_1_gram][word]/n_1_gram_sum

    for key, value in n_gram_probs.iteritems():
        n_gram_probs[key] = sorted(value.items(), key=operator.itemgetter(1), reverse=True)

    print len(n_gram_probs)
    return n_gram_probs

def get_start_key(dataset,label,n=5):
    cnt = 0    
    probs = []
    sentences = []    
    
    for speech in dataset[label]:
        sent = speech[1]
        words = sent.split()[:n-1]
        start = " ".join(words)
        
        cnt+=1
        
        if start in sentences:
            idx = sentences.index(start)
            probs[idx] +=1
        else:
            sentences.append(start)
            probs.append(1)
            
    for i in range(len(probs)):
        probs[i] = probs[i] / cnt  
    
    idx = np.random.multinomial(1, probs)[0]    
    result = START_OF_SPEECH + " " + sentences[idx]
    result = tuple(result.split())
    
    return result


def get_word_prob_for_topics(dataset, c, word, topics):
    count = 0.0
    totlen = 0.001
    for speech in dataset[c]:
        full_speech = " ".join(speech)
        speech_prob = 0
        for t,prob in topics.iteritems():            
            if t in full_speech:
                speech_prob += prob
        
        if speech_prob > 0.0:        
            count+=full_speech.count(word)*speech_prob
            totlen += len(full_speech.split())*speech_prob

    p_w = count/totlen
    return p_w

def get_n_topics_from_ngram(dataset, jk_trend,jk, c, ngram, n=3):
    topics = defaultdict(float)
    ngram_key = " ".join(ngram)
    for speech in dataset[c]:
        full_speech = " ".join(speech)

        if ngram_key in full_speech:
            for key in jk_trend[c].keys():
                topics[key] += full_speech.count(key)

    for key,cnt in  topics.iteritems():
        topics[key] = cnt/jk[c][key]
    result = []
    for t in sorted(topics.items(), key=operator.itemgetter(1),reverse=True)[:n]:
        result.append(t[0])
    return result

def get_topics_from_speech(speech, jk_trend,jk, c, n=3):
    
    topics = defaultdict(float)
    for key in jk_trend[c].keys():
        if key in speech:
            topics[key] += speech.count(key)            

    for key,cnt in  topics.iteritems():
        topics[key] = cnt/jk[c][key]
        
    if n is None:
        n=len(topics)
    result = dict()
    sorted_topics = sorted(topics.items(), key=operator.itemgetter(1),reverse=True)[:n]
    for t in sorted_topics:
        result[t[0]] = t[1]/sum([pair[1] for pair in sorted_topics])
    return result


import pickle
def create_corpus_pos_tags(dataset):
    all_pos_tags = set()
    for label,speeches in dataset.iteritems():
        print label,'...',
        sys.stdout.flush()
        for sp in speeches:            
            for sent in sp[1:-1]:
                tags = pos_tag(sent.split()[:-1])
                tag_sequence = [x[1] for x in tags]
                tag_sequence = " ".join(tag_sequence)                
                all_pos_tags.add(tag_sequence)
        print 'Done!'
        sys.stdout.flush()
    pickle.dump( all_pos_tags, open( "all_pos_tags.p", "wb" ) )
    return all_pos_tags

def evaluate_grammar(speech,verbose=True):
    sp = speech.replace(START_OF_SPEECH,'')
    sp = sp.replace(END_OF_SPEECH,'')
    sentences = sp.split(END_OF_SENTENCE)
    if len(sentences[-1].strip())== 0:
        sentences = sentences[:-1]
    
    
    all_pos_tags = pickle.load( open( "all_pos_tags.p", "rb" ) )
               

    acc_cnt = 0
    for sent in sentences:
        tags = pos_tag(sent.split()) 
        tag_sequence = [x[1] for x in tags]
        tag_sequence = " ".join(tag_sequence)
        
        if tag_sequence in all_pos_tags:
            acc_cnt += 1
        elif verbose == True:
            print sent
        
    
    return acc_cnt/len(sentences)

def evaluate_content(gen_speech, dataset, label,jk,jk_trend):
    gen_topics = get_topics_from_speech(gen_speech, jk_trend,jk, label, n=None)
    sorted_gen_topics = sorted(gen_topics.items(), key=operator.itemgetter(1),reverse=True)
    num_topics = len(sorted_gen_topics)
    if num_topics==0:
        return 1.0

    max_cnt = 0

    for speech in dataset[label]:
        sp = " ".join(speech)
        topics = get_topics_from_speech(sp, jk_trend,jk, label, n=num_topics)
        sorted_topics = sorted(topics.items(), key=operator.itemgetter(1),reverse=True)
        sorted_topics = [t[0] for t in sorted_topics]

        cnt =0
        for i in range(num_topics):
            if i < len(sorted_topics):
                if sorted_topics[i] == sorted_gen_topics[i][0]:
                    cnt += sorted_gen_topics[i][1]
                    
        if cnt > max_cnt:
            max_cnt = cnt
    return max_cnt

def generate_speech_wba(dataset,n_gram_probs,ngram_class_probs,corpus_ngram_props,jk_trend,jk,label,lamb=0.3,max_words=900):
    wordcnt = 0
    next_word = ''
    nn = len(n_gram_probs[label].keys()[0])
    tuple_key = get_start_key(dataset,label,n=nn)
    print " ".join(tuple_key),
    my_speech = " ".join(tuple_key)
    current_sentence = my_speech
    topic_cnt = defaultdict(float)

    #all_pos_tags = pickle.load( open( "all_pos_tags.p", "rb" ) )
    
    sen_count = 0
    topics = []
    speech_sentences = []
    while (next_word != END_OF_SPEECH) and ((wordcnt < max_words) or (next_word != END_OF_SENTENCE)):
       
        topics = get_topics_from_speech(my_speech,jk_trend,jk,label)
        
        words = []
        probs = []
        topic_probs = dict()       
        
        for (word,ngram_prob) in n_gram_probs[label][tuple_key]:
            topic_prob = get_word_prob_for_topics(dataset,label,word,topics)
            topic_probs[word] = topic_prob
            
        sum_probs = sum(topic_probs.values())
        if sum_probs > 0:
            for word,prob in topic_probs.iteritems():
                topic_probs[word] = topic_probs[word]/sum_probs
        

        for (word,ngram_prob) in n_gram_probs[label][tuple_key]:
        #for (word,ngram_prob) in corpus_ngram_props[tuple_key]:
            topic_prob = topic_probs[word]   
            lang_prob = ngram_prob            
            
            prob = lamb*lang_prob + (1-lamb)*topic_prob
            phrase = " ".join(tuple_key) + ' ' + word
            prob = prob/(1+my_speech.count(phrase)**2)
                
            if prob <= 0:
                continue 
           
            words.append(word)
            probs.append(prob)
            
        if len(probs) > 1:   
            probs = [p/sum(probs) for p in probs]
            ni = np.random.multinomial(1, probs)[0] 
        else:
            ni = 0
            
        if len(word) > 0:
            next_word = words[ni]
        else:
            next_word = END_OF_SENTENCE

        if next_word == END_OF_SENTENCE:
            print '.'
            speech_sentences.append(current_sentence)
            current_sentence = ''
            sen_count += 1
        else:
            print next_word,
            current_sentence = current_sentence + ' ' + next_word
        my_speech = my_speech + ' ' + next_word
        tuple_key = tuple_key[1:] + (next_word,)
        wordcnt += 1
    return my_speech