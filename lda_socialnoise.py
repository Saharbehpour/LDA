# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 10:46:37 2019

@author: Monab
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.models.ldamodel import LdaModel

if __name__=='__main__':
# stopwords list
    stops = []
    fp = open("stopwords.txt", "r")
    stop = fp.readlines()
    fp.close
    for line in stop:
        stops.append(line.strip("\n")) 
#print(stops)
    fileopen = open("cheltman.csv", encoding = 'unicode_escape')
    lines = fileopen.readlines()
#print(lines)
    corpus = []
    punc = string.punctuation # get English punctuations by string
    punc = '[,.!\'"#$%&()*+-/:;<=>?@`_~{}|0,1,2,3,4,5,6,7,8,9]'
    for line in lines:
        line = re.sub(punc,'',line).strip("\n")
        corpus.append(line)
#print(corpus)
# using the english stopword list
    vectorizer = CountVectorizer(lowercase=True, stop_words='english',max_df=0.50,min_df=1,ngram_range=(1, 3),max_features=20000)
    #vectorizer = CountVectorizer(lowercase=True, stop_words=frozenset(stops),max_df=0.50,min_df=0.01,ngram_range=(1, 3),max_features=20000)
    vector = vectorizer.fit_transform(corpus)
    feature_name = vectorizer.get_feature_names()
    dictionary = {}

    for ind in range(len(feature_name)):
        dictionary[ind] = feature_name[ind]
        corpus_array = vector.toarray()
        corpus = []
        n_doc, n_vocab = np.shape(corpus_array)
    for doc_index in range(n_doc):
        doc = []
    for word_index in range(n_vocab):
        word_freq = corpus_array[doc_index][word_index]
        if word_freq != 0:
           doc.append((word_index, word_freq)) 
    corpus.append(doc)
  
### dictionary mapping
    np.random.seed(40)
    lda = LdaModel(corpus, id2word=dictionary, num_topics=50, update_every=1, passes=3) # TO ADJUST the parameters, mainly num_topics which is number of topics
    results = lda.print_topics(num_topics=20, num_words=5)
    for result in results:
        print(result)
#
#
#
##
#
