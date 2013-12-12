# -*- coding: utf-8 -*-
'''
Common utility methods for key concept extraction.
'''
from nltk.corpus import stopwords as nltk_stopwords
import unicodedata

def ascii_approximation(s):
    """
    Returns an ascii "approximation" to the given unicode string.
        @param s: The string to be approximated.
    """
    if not isinstance(s, unicode): s = str(s).decode('utf-8', 'replace')
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'replace')

def wash_string(text):
  text = text.replace(")", "")
  text = text.replace("(", "")
  return text.strip()

def default_stopwords():
    '''
    Returns a default set of stopwords.
    '''
    stopwords = nltk_stopwords.words('english')
    stopwords.extend(['%', "''", "'m", '--', '``', '|', '/', '-', '.', '?', '!', ',', ';', ':', '(', ')', '"', "'", '`', '’', "'s"])
    return set(stopwords)
DEFAULT_STOPWORDS = default_stopwords() 

MONTHS = set([
    'january', 'jan.', 'jan',
    'february', 'feb.', 'feb',
    'march', 'mar.', 'mar',
    'april', 'apr.', 'apr',
    'may',
    'june', 'jun.', 'jun',
    'july', 'jul.', 'jul',
    'august', 'aug.', 'aug',
    'september', 'sep.', 'sept.', 'sep', 'sept',
    'october', 'oct.', 'oct',
    'december', 'dec.', 'dec',
])
def post_process(phrases):
    """
    Returns a version of the given collection of phrases with
    stop words removed from the beginning and end of phrases
    and omitting phrases that are too long.
        @param phrases: The phrases to process.
    """
    filtered_phrases = set([])
    for phrase in phrases:
        phrase, length = trim(phrase)
        if length > 4 or length == 0 or phrase.lower() in MONTHS: continue
        filtered_phrases.add(phrase)
    return filtered_phrases

def sub_leaves(tree, node):
    '''
    Extracts and returns leaf nodes in the given tree below the given node.
    '''
    return [t.leaves() for t in tree.subtrees(lambda s: s.node == node)]

import nltk.data
SENTENCE_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
def tag_document(document):
  tokenized_sentences = [] 
  tagged_sentences = []
  for sentence in SENTENCE_DETECTOR.tokenize(document.strip()):
    #tokens = [ascii_approximation(s) for s in nltk.word_tokenize(sentence)]
    tokens = [wash_string(token) for token in nltk.word_tokenize(sentence) if wash_string(token)]
    tags = nltk.pos_tag(tokens)
    tokenized_sentences.append(tokens)
    tagged_sentences.append(tags)
  return tokenized_sentences, tagged_sentences

def trim(phrase):      
    '''
    Returns a version of the given phrase with stop words removed from the 
    beginning and end of the phrase.
        @param phrase: The phrase to trim.
    '''
    words = phrase.split()
    while words and words[0] in DEFAULT_STOPWORDS:
        words.pop(0)
    while words and words[-1] in DEFAULT_STOPWORDS:
        words.pop(-1)   
    return " ".join(words), len(words)       
