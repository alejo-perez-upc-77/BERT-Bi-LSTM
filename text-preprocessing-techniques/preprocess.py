""" Copyright 2017, Dimitrios Effrosynidis, All rights reserved. """

from time import time
import numpy as np
import string

from techniques import *

#print("Starting preprocess..\n")

""" Tokenizes a text to its words, removes and replaces some of them """    
finalTokens = [] # all tokens
stoplist = stopwords.words('english')
lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer

def tokenize(text): #wordCountBefore, textID, y):
    # totalAdjectives = 0
    # totalAdverbs = 0
    # totalVerbs = 0
    # onlyOneSentenceTokens = [] # tokens of one sentence each time

    final_tokens = []
    tokens = nltk.word_tokenize(text)
    tokens = replaceNegations(tokens) # Technique 6: finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator) # Technique 7: remove punctuation
    
    tokens = nltk.word_tokenize(text) # it takes a text as an input and provides a list of every token in it

### NO POS TAGGING BEGIN (If you don't want to use POS Tagging keep this section uncommented) ###
    for w in tokens:
      
      if (w not in stoplist): # Technique 10: remove stopwords
        final_word = addCapTag(w) # Technique 8: Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_
        final_word = final_word.lower() # Technique 9: lowercases all characters
        final_word = replaceElongated(final_word) # Technique 11: replaces an elongated word with its basic form, unless the word exists in the lexicon
        # if len(final_word)>1:
        #     final_word = spellCorrection(final_word) # Technique 12: correction of spelling errors
        #     print("if its the case correction",final_word)
        final_word = lemmatizer.lemmatize(final_word) # Technique 14: lemmatizes words
        final_word = stemmer.stem(final_word) # Technique 15: apply stemming to words
        final_tokens.append(str(final_word))

        
    return final_tokens



