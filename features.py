#First, we import all the packcages for the processing
import nltk
import re
import pandas
import numpy

from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from nltk.metrics import jaccard_distance

from scipy.stats import pearsonr

from utils import wordnet_pos_code
from utils import lemmatize
from utils import lemmatize_sentence
from utils import filter_for_wordnet

from scipy.spatial.distance import cosine





# __________________ FEATURES _________________________________________



def extract_absolute_difference(sentence1, sentence2):
    """Diff in {all tokens, adjectives, adverbs, nouns, and verbs}"""
    s1, s2 = nltk.word_tokenize(sentence1), nltk.word_tokenize(sentence2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = abs(len(s1) - len(s2)) / float(len(s1) + len(s2))
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = abs(cnt1 - cnt2) / float(cnt1 + cnt2)
    return t1, t2, t3, t4, t5

def extract_mmr_t(s1, s2):
    """Common in {all tokens, adjectives, adverbs, nouns, and verbs}"""
    shorter = 1
    if(len(s1) > len(s2)):  shorter = 2

    s1, s2 = nltk.word_tokenize(s1), nltk.word_tokenize(s2)
    pos1, pos2 = pos_tag(s1), pos_tag(s2)
    # all tokens
    t1 = (len(s1)+0.001) / (len(s2) +0.001)
    # all adjectives
    cnt1 = len([1 for item in pos1 if item[1].startswith('J')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('J')])
    if cnt1 == 0 and cnt2 == 0:
        t2 = 0
    else:
        t2 = (cnt1 +0.001) / (cnt2 + 0.001)
    # all adverbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('R')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('R')])
    if cnt1 == 0 and cnt2 == 0:
        t3 = 0
    else:
        t3 = (cnt1 +0.001) / (cnt2+0.001)
    # all nouns
    cnt1 = len([1 for item in pos1 if item[1].startswith('N')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('N')])
    if cnt1 == 0 and cnt2 == 0:
        t4 = 0
    else:
        t4 = (cnt1 +0.001) / (cnt2 +0.001)
    # all verbs
    cnt1 = len([1 for item in pos1 if item[1].startswith('V')])
    cnt2 = len([1 for item in pos2 if item[1].startswith('V')])
    if cnt1 == 0 and cnt2 == 0:
        t5 = 0
    else:
        t5 = (cnt1+ 0.001) / (cnt2 + 0.001)

    if shorter == 2:
        t1 = 1 / (t1 + 0.001)
        t2 = 1 / (t2 + 0.001)
        t3 = 1 / (t3 + 0.001)
        t4 = 1 / (t4 + 0.001)
        t5 = 1 / (t5 + 0.001)

    return [t1, t2, t3, t4, t5]    
