import nltk
import string
import pandas
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def lemmatize(p, wordnet_lemmatized):
    if p[1][0] in {'N','V','A','R'}:
        return wordnet_lemmatized.lemmatize(p[0].lower(), pos=p[1][0].lower())
    return p[0]

def filter_for_wordnet(tagged_sentence):
    return [(word, pos_tag) for (word, pos_tag) in  tagged_sentence if pos_tag.startswith('NN') or 
                                                                      pos_tag.startswith('VB') or 
                                                                      pos_tag.startswith('RB') or 
                                                                      pos_tag.startswith('JJ')]

def lemmatize_sentence(sentence, filter_words=True):
    # Tokenize by sentences into words in lower case 
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    # Generate word post tags
    tagged_sentence = pos_tag(tokenized_sentence) # [ (word, label), (word2, label2) ...]
    
    if filter_words:
        tagged_sentence = filter_for_wordnet(tagged_sentence)
    lemmas_sentence = [lemmatize(tagged_word, wnl) for tagged_word in tagged_sentence] 
    
    return lemmas_sentence, tagged_sentence

def wordnet_pos_code(tag):
    '''
    Convert NLTK POS tags to WordNet standards 
    @ param tag NLTK POS tag 
    '''
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        print("Error, unknown tag: " + tag)
        return ''

def load_gold_standard_dataset(): 
    # !!!! NOTE: DO not change the order of this list --- The labels are ordered consequently !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    test_files = ['STS.input.MSRvid.txt', 'STS.input.SMTeuroparl.txt', 'STS.input.surprise.OnWN.txt', 'STS.input.surprise.SMTnews.txt'] 
    ground_truth_files = ['STS.gs.MSRvid.txt', 'STS.gs.SMTeuroparl.txt', 'STS.gs.surprise.OnWN.txt', 'STS.gs.surprise.SMTnews.txt' ]
    file_path = 'test-gold/'

    # Load sentences ------------------------------------------------------------------------------------------
    test_set = pandas.DataFrame(columns=['S1', 'S2'])
    for test_file in test_files: 
        source = pandas.read_csv(file_path + test_file, sep='\t', lineterminator='\n', names=['S1','S2'])
        test_set = test_set.append(source)
        #print('%s - %s' % (test_file, s))
    # Reset indexes
    test_set.reset_index(inplace=True)
    test_set.drop(columns=['index'], inplace=True)

    # Preprocess 
    pre_process_dataset(test_set)
    print('Gold standard set contains %s sentences' % (test_set.shape,))

    # Load sentences labels ------------------------------------------------------------------------------------ 
    test_set_labels = pandas.DataFrame(columns=['Label'])
    for labels_file in ground_truth_files: 
        source = pandas.read_csv(file_path + labels_file, sep='\t', lineterminator='\n', names=['Label'])
        s = source.shape
        test_set_labels = test_set_labels.append(source)
        #print('%s - %s' % (labels_file, s))
    # Reset indexes
    test_set_labels.reset_index(inplace=True)
    test_set_labels.drop(columns=['index'], inplace=True)
    
    # Concatenate labels with sentences
    test_set = pandas.concat([test_set,test_set_labels], axis=1)


    print('Gold standard set contains %s labeles' % (test_set_labels.shape,))
    
    return test_set

def pre_process_dataset(source):
    for index, db_entry in source.iterrows():
        s1, s2 = db_entry['S1'], db_entry['S2'] 
        # Remove punctuation
        s1 = s1.translate(str.maketrans('', '', string.punctuation))
        s2 = s2.translate(str.maketrans('', '', string.punctuation))
        source.at[index, 'S1'] = s1
        source.at[index, 'S2'] = s2


def load_dataset():  
    # Load sentences ------------------------------------------------------------------------------------------   
    X_train = pandas.read_csv("data/en-train.txt", sep='\t', lineterminator='\n', names=['S1','S2','Label'])
    X_val = pandas.read_csv("data/en-val.txt", sep='\t', lineterminator='\n', names=['S1','S2','Label'])
    X_test = pandas.read_csv("data/en-test.txt", sep='\t', lineterminator='\n', names=['S1','S2','Label'])

    pre_process_dataset(X_train)
    pre_process_dataset(X_val)
    pre_process_dataset(X_test)

    print("Training set ", X_train.shape)
    print("Validation set ", X_val.shape)
    print("Test set ", X_test.shape)

    return X_train, X_val, X_test

