#First, we import all the packcages for the processing
import nltk
import re
import pandas
import numpy 
import difflib
import time 

import gensim
import os
import shutil
import hashlib
import editdistance
from pyjarowinkler import distance

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from nltk.metrics import jaccard_distance
from nltk.corpus import wordnet_ic

from scipy.stats import pearsonr

from utils import wordnet_pos_code
from utils import lemmatize
from utils import lemmatize_sentence
from utils import filter_for_wordnet

from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from nltk.parse import corenlp as nlp



wnl = WordNetLemmatizer()

NE_tagger = nlp.CoreNLPParser(url='http://localhost:9000', tagtype="ner")

from nltk.parse.corenlp import CoreNLPDependencyParser
parser = CoreNLPDependencyParser(url='http://localhost:9000')

brown_ic = wordnet_ic.ic('ic-brown.dat')
stop_words = stopwords.words('english')

glove_model = None

# vec1 = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
# vec2 = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

#____________ UTILS ______________________________________________
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def load_glove(filename="glove.6B.300d.txt"):
    def getFileLineNums(filename):
        f = open(filename, 'r', encoding="utf8")
        count = 0
        for line in f:
            count += 1
        return count

    def prepend_slow(infile, outfile, line):
        with open(infile, 'r', encoding="utf8") as fin:
            with open(outfile, 'w', encoding="utf8") as fout:
                fout.write(line + "\n")
                for line in fin:
                    fout.write(line)

    def load(filename):
        start_time = time.time()
        print("Loaging glove model: %s ..." % filename, end='')
        num_lines = getFileLineNums(filename)
        gensim_file = 'glove/glove_model_gensim.txt'
        gensim_first_line = "{} {}".format(num_lines, 300)
        # Prepends the line.        
        prepend_slow(filename, gensim_file, gensim_first_line)
        model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
        elapsed_time = time.time() - start_time
        print(" took : %.5f" % elapsed_time)
        return model

    return load("glove/" + filename)

def ner_transform(sentence):
    tagged_sent = NE_tagger.tag(sentence.split())
    new_sentence = []
    for word, tag in tagged_sent:
        if tag == 'O':
            new_sentence.append(word.lower())
        else:
            new_sentence.append(tag)
            print(word, '-' , tag, '/', end='')
    # print(sentence)
    # print(new_sentence)
    return new_sentence

# ___________ SIMILARITIES ___________________________________________       
def jaccard_similarity(s1, s2):
    try:
        tokenized_sentence_1 = nltk.word_tokenize(s1.lower())
        tokenized_sentence_2 = nltk.word_tokenize(s2.lower())
    except:
        print("Error: S1[%s] \n S2[%s]" % (s1, s2))
        return 0
    # Compute similarity
    if len(tokenized_sentence_1) > 0 and len(tokenized_sentence_2) > 0:
        similarity = 1- jaccard_distance(set(tokenized_sentence_1), set(tokenized_sentence_2))
        return similarity
    else:
        return 0

def ne_simmilarity(s1, s2):
    sent1 = ner_transform(s1)
    sent2 = ner_transform(s2)
    # Compute similarity
    if len(sent1) > 0 and len(sent2) > 0:
        similarity = 1 - jaccard_distance(set(sent1), set(sent2))
        # Compute label of similarity 
        return similarity
    else:
        return 0

def edit_distance(s1, s2):
    normalizer = len(s1) if len(s1) > len(s2) else len(s2)
    return editdistance.eval(s1, s2) / normalizer

def pyjarowinkler_distance(s1,s2):
    return distance.get_jaro_distance(s1, s2, winkler=True, scaling=0.1)

def hamming_similarity(s1, s2):
    # max_len = len(s1) if len(s1) > len(s)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def ngrams_similarity(s1, s2, filter_stop_words=True):
     # Tokenize by sentences into words in lower case 
    tokenized_sentence_1 = nltk.word_tokenize(s1.lower())
    tokenized_sentence_2 = nltk.word_tokenize(s2.lower())
    
    if filter_stop_words:
        tokenized_sentence_1 = [token for token in tokenized_sentence_1 if token not in stop_words]
        tokenized_sentence_2 = [token for token in tokenized_sentence_2 if token not in stop_words]
    
    grams_lst_1 = [w for w in nltk.ngrams(tokenized_sentence_1, 2)]
    grams_lst_2 = [w for w in nltk.ngrams(tokenized_sentence_2, 2)]
    if len(grams_lst_1) > 0 and len(grams_lst_2) > 0:
        sim2 = 1 - jaccard_distance(set(grams_lst_1), set(grams_lst_2))
    else:
        sim2 = 0

    grams_lst_1 = [w for w in nltk.ngrams(tokenized_sentence_1, 3)]
    grams_lst_2 = [w for w in nltk.ngrams(tokenized_sentence_2, 3)]
    if len(grams_lst_1) > 0 and len(grams_lst_2) > 0:
        sim3 = 1 - jaccard_distance(set(grams_lst_1), set(grams_lst_2))
    else:
        sim3 = 0

    grams_lst_1 = [w for w in nltk.ngrams(tokenized_sentence_1, 4)]
    grams_lst_2 = [w for w in nltk.ngrams(tokenized_sentence_2, 4)]
    if len(grams_lst_1) > 0 and len(grams_lst_2) > 0:
        sim4 = 1 - jaccard_distance(set(grams_lst_1), set(grams_lst_2))
    else:
        sim4 = 0

    return sim2, sim3, sim4

def lemmas_similarity(s1, s2, filter_stop_words=True):
    """
    Jaccard lematized sentences similarity 
    """
    # Tokenize by sentences into words in lower case 
    tokenized_sentence_1 = nltk.word_tokenize(s1.lower())
    tokenized_sentence_2 = nltk.word_tokenize(s2.lower())
    
    if not filter_stop_words:
        tokenized_sentence_1 = [token for token in tokenized_sentence_1 if token not in stop_words]
        tokenized_sentence_2 = [token for token in tokenized_sentence_2 if token not in stop_words]
    
    tagged_sentence_1 = pos_tag(tokenized_sentence_1) # [ (word, POS_TAG), ...]
    tagged_sentence_2 = pos_tag(tokenized_sentence_2) # [ (word, POS_TAG), ...]
    
    lemmas_sentence_1 = [lemmatize(tagged_word, wnl) for tagged_word in tagged_sentence_1] 
    lemmas_sentence_2 = [lemmatize(tagged_word, wnl) for tagged_word in tagged_sentence_2] # [LEMMA_1, ...]
    
    # Compute similarity
    if len(lemmas_sentence_1) > 0 and len(lemmas_sentence_2) > 0:
        similarity = 1 - jaccard_distance(set(lemmas_sentence_1), set(lemmas_sentence_2))
        # Compute label of similarity 
        return similarity
    else:
        return 0

def information_content_similarity(s1, s2):
    """ 
    Compute the sentence similairty using information content from wordnet
    (words are disambiguated first to Synsets by means of Lesk algorithm) 
    """
    lemmas_sentence_1, tagged_sentence_1 = lemmatize_sentence(s1.lower())
    lemmas_sentence_2, tagged_sentence_2 = lemmatize_sentence(s2.lower())

    # Disambiguate words and create list of sysnsets 
    synsets_sentence_1 = []
    for (lemma, word_tag) in zip(lemmas_sentence_1, tagged_sentence_1):
        synset = lesk(lemmas_sentence_1, lemma, wordnet_pos_code(word_tag[1]))
        if synset is not None:
            synsets_sentence_1.append(synset)
        else:
            found = wordnet.synsets(lemma, wordnet_pos_code(word_tag[1]))
            if len(found) > 0:
                synsets_sentence_1.append(found[0]) 
                #print("Warn: lemma [%s] returned no disambiguation...using synset : %s" % (lemma, found[0])) 
    synsets_sentence_2 = []
    for (lemma, word_tag) in zip(lemmas_sentence_2, tagged_sentence_2):
        synset = lesk(lemmas_sentence_2, lemma, wordnet_pos_code(word_tag[1]))
        if synset is not None:
            synsets_sentence_2.append(synset)
        else:
            found = wordnet.synsets(lemma, wordnet_pos_code(word_tag[1]))
            if len(found) > 0:
                synsets_sentence_2.append(found[0]) 
                #print("Warn: lemma [%s] returned no disambiguation...using synset : %s" % (lemma, found[0])) 

    score, count = 0.0, 0
    # For each word in the first sentence
    for synset in synsets_sentence_1:
        L = []
        for ss in synsets_sentence_2:
            try:
                L.append(synset.lin_similarity(ss, brown_ic))
            except:
                continue
        if L: 
            best_score = max(L)
            score += best_score
            count += 1
    # Average the values
    if count > 0: score /= count
    return score

def simple_baseline_similarity(s1, s2):
    """
    Find the sequence similarity between two words considering lemmas and words
    """
    # Tokenize by sentences into words in lower case 
    tokenized_sentence_1 = nltk.word_tokenize(s1.lower())
    tokenized_sentence_2 = nltk.word_tokenize(s2.lower())

    tagged_sentence_1 = pos_tag(tokenized_sentence_1) # [ (word, POS_TAG), ...]
    tagged_sentence_2 = pos_tag(tokenized_sentence_2) # [ (word, POS_TAG), ...]
    
    lemmas_sentence_1 = [lemmatize(tagged_word, wnl) for tagged_word in tagged_sentence_1 if not tagged_word in stop_words] 
    lemmas_sentence_2 = [lemmatize(tagged_word, wnl) for tagged_word in tagged_sentence_2 if not tagged_word in stop_words] # [LEMMA_1, ...]
    
    word_seq_match = difflib.SequenceMatcher(None, tokenized_sentence_1, tokenized_sentence_2)
    word_match = word_seq_match.find_longest_match(0, len(tokenized_sentence_1), 0, len(tokenized_sentence_2))

    lemm_seq_match = difflib.SequenceMatcher(None, lemmas_sentence_1, lemmas_sentence_2)
    lemm_match = lemm_seq_match.find_longest_match(0, len(lemmas_sentence_1), 0, len(lemmas_sentence_2))

    word_sim = word_match.size/(max(len(tokenized_sentence_1), len(tokenized_sentence_2)) + 0.001)
    lemm_sim = lemm_match.size/(max(len(lemmas_sentence_1), len(lemmas_sentence_2)) + 0.001)

    return word_sim, lemm_sim

def dependency_similarity(s1, s2):
    """
    Find the jaccard similarity between the semantic depency parsing nodes of the sentences
    using CoreNLP dependency parser.
    """
    # pass
    parsed_sentence_1 = parser.raw_parse(s1)
    parsed_sentence_2 = parser.raw_parse(s2)
        
    tree1 = next(parsed_sentence_1)
    tree2 = next(parsed_sentence_2)
        
    triples1 = [t for t in tree1.triples()]
    triples2 = [t for t in tree2.triples()] 

    # Compute similarity
    if len(triples1) != 0 and len(triples2) != 0:
        similarity = 1 - jaccard_distance(set(triples1), set(triples2))
        return similarity
    else:
        return 0

def synsets_similarity(s1, s2):
    """
    Find the jaccard similarity between two sentences synsets using lesk algorithm
    to disambiguate words given their context.
    """
    lemmas_sentence_1, tagged_sentence_1 = lemmatize_sentence(s1.lower())
    lemmas_sentence_2, tagged_sentence_2 = lemmatize_sentence(s2.lower())

    # Disambiguate words and create list of sysnsets 
    synsets_sentence_1 = []
    for (lemma, word_tag) in zip(lemmas_sentence_1, tagged_sentence_1):
        if lemma in stop_words:
            continue
        synset = lesk(lemmas_sentence_1, lemma, wordnet_pos_code(word_tag[1]))
        if synset is not None:
            synsets_sentence_1.append(synset)
        else:
            found = wordnet.synsets(lemma, wordnet_pos_code(word_tag[1]))
            if len(found) > 0:
                synsets_sentence_1.append(found[0]) 
                #print("Warn: lemma [%s] returned no disambiguation...using synset : %s" % (lemma, found[0])) 

    synsets_sentence_2 = []
    for (lemma, word_tag) in zip(lemmas_sentence_2, tagged_sentence_2):
        if lemma in stop_words:
            continue
        synset = lesk(lemmas_sentence_2, lemma, wordnet_pos_code(word_tag[1]))
        if synset is not None:
            synsets_sentence_2.append(synset)
        else:
            found = wordnet.synsets(lemma, wordnet_pos_code(word_tag[1]))
            if len(found) > 0:
                synsets_sentence_2.append(found[0]) 
                #print("Warn: lemma [%s] returned no disambiguation...using synset : %s" % (lemma, found[0])) 

    # Compute similarity
    if len(synsets_sentence_1) != 0 and len(synsets_sentence_2) != 0:
        similarity = 1 - jaccard_distance(set(synsets_sentence_1), set(synsets_sentence_2))
        return similarity
    else:
        return 0 

def get_sentence_mean_vec(sentence):
    """
    Provided a sentence of words, find the sentence embedding vector representation
    using the mean vector from all of the words embedding vector representations.
    """
    sentence_vecs = numpy.array([])
    
    sent1 = nltk.word_tokenize(sentence)
    for w in sent1: 
        w = w.strip("'?.,- ")
        if not w in stop_words and w.lower() in glove_model:
            word_vec = numpy.array([glove_model[w.lower()]])
            if sentence_vecs.shape[0] == 0: # Initialize sentence vectors
                sentence_vecs = word_vec
            else:
                sentence_vecs = numpy.vstack((sentence_vecs, word_vec))
    # print(sentence_vecs.shape)
    if sentence_vecs.shape[0] == 0:
        return None
    elif sentence_vecs.shape == (300,):
        return numpy.expand_dims(sentence_vecs, axis=0)
    return numpy.mean(sentence_vecs, axis=0)

def glove_word2vec_vec_similarity(s1, s2):
    global glove_model

    if glove_model is None:
        glove_model = load_glove()

    s1_vec = get_sentence_mean_vec(s1)
    s2_vec = get_sentence_mean_vec(s2)

    if s1_vec is None or s2_vec is None:
        return 0
    ret = numpy.dot(s1_vec, s2_vec) / (numpy.linalg.norm(s1_vec) * numpy.linalg.norm(s2_vec))
    ret = 5*(ret + 1) / 2
    return ret


def longest_common_subsequence(s1, s2): 

    lemmas_sentence_1, _ = lemmatize_sentence(s1.lower())
    lemmas_sentence_2, _ = lemmatize_sentence(s2.lower())
    sent1 = [w for w in lemmas_sentence_1 if not w in stop_words]
    sent2 = [w for w in lemmas_sentence_2 if not w in stop_words]

    ss1 = ' '.join(sent1)
    ss2 = ' '.join(sent2)
    m = len(ss1) 
    n = len(ss2) 
    
    if m == 0 or n ==0:
        return 0
    # declaring the array for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif ss1[i-1] == ss2[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    normalizer = len(ss1) if len(ss1) < len(ss2) else len(ss2)
    
    return L[m][n] / normalizer
    
def extract_overlap_pen(s1, s2):
    """
    :param s1:
    :param s2:
    :return: overlap_pen score
    """
    lemmas_sentence_1, _ = lemmatize_sentence(s1.lower())
    lemmas_sentence_2, _ = lemmatize_sentence(s2.lower())
    ss1 = [w for w in lemmas_sentence_1 if not w in stop_words]
    ss2 = [w for w in lemmas_sentence_2 if not w in stop_words]

    ovlp_cnt = 0
    for w1 in ss1:
        ovlp_cnt += ss2.count(w1)
    score = 2 * ovlp_cnt / (len(ss1) + len(ss2) + .001)
    return score


# def sif_embeddings(sentences, alpha=1e-3):
#     """Compute the SIF embeddings for a list of sentences
#     Parameters
#     ----------
#     sentences : list
#         The sentences to compute the embeddings for
#     model : `~gensim.models.base_any2vec.BaseAny2VecModel`
#         A gensim model that contains the word vectors and the vocabulary
#     alpha : float, optional
#         Parameter which is used to weigh each individual word based on its probability p(w).
#     Returns
#     -------
#     numpy.ndarray 
#         SIF sentence embedding matrix of dim len(sentences) * dimension
#     """
#     global glove_model
    
#     vlookup = glove_model.wv.vocab  # Gives us access to word index and count
#     vectors = glove_model.wv        # Gives us access to word vectors
#     size = glove_model.vector_size  # Embedding size
    
#     Z = 0
#     for k in vlookup:
#         Z += vlookup[k].count # Compute the normalization constant Z
    
#     output = []
    
#     # Iterate all sentences
#     for s in sentences:
#         count = 0
#         v = numpy.zeros(size, dtype=REAL) # Summary vector
#         # Iterare all words
#         for w in s:
#             # A word must be present in the vocabulary
#             if w in vlookup:
#                 for i in range(size):
#                     v[i] += ( alpha / (alpha + (vlookup[w].count / Z))) * vectors[w][i]
#                 count += 1 
                
#         if count > 0:
#             for i in range(size):
#                 v[i] *= 1/count
#         output.append(v)
#     return numpy.vstack(output)