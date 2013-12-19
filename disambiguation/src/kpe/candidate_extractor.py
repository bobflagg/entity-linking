# -*- coding: UTF-8 -*-
'''
Tools for collecting candidate key concepts from a POS tagged document.
'''
from nltk.chunk import RegexpParser
from nltk.probability import FreqDist
import re
from util import sub_leaves

SINGLE_WORD_FREQ_CUT_OFF = 6
PATTERNS = r'''
    NP:  {<CD|VBN>?<NN.*|JJ.*>*<CD>?<NN.*|VBG><CD>?}
'''
PATTERNS_X = r'''
    NP:  {<NN.*|JJ.*|CD>*<NN.*|VBG><CD>?}
         {<NN.*|JJ.*>*<CD>?<NN.*|VBG><CD>?}
'''
PATTERNS_ALT = r'''
    NP:  {<NN.*|JJ.*>*<NN.*><CC><NN.*|VBG><CD>?}
         {<NN.*|JJ.*>*<CD>?<NN.*|VBG><CD>?}
'''
# ('2009', 'CD'), ('Grammy', 'NNP'), ('Awards', 'NNS')
NP_CHUNCKER = RegexpParser(PATTERNS)
EARLY_CANDIDATE_CUTOFF = 25
LATE_CANDIDATE_CUTOFF = 10

def extract_candidates(tagged_sentences):
    '''
    Returns three lists:
        - the candidate key concepts of the given document;
        - the candidate key concepts occurring early in the given document; and 
        - the candidate key concepts occurring late in the given document.
        @param tagged_sentences: The POS tagged document.    
    '''
    #print tagged_sentences
    candidates = []
    early= set([])
    late= set([])
    num_sentences = len(tagged_sentences)
    pos = 0
    for tagged_sentence in tagged_sentences:
        pos += 1
        ts = contract_pos(tagged_sentence)
        if len(ts) > 0:
            tree = NP_CHUNCKER.parse(ts)
            for leaf in sub_leaves(tree, 'NP'):
                candidate = ' '.join(word for word, _ in leaf)   
                candidate = remove_time_ref(candidate)   
                candidate = remove_quantity_ref(candidate)                  
                if valid_candidate(candidate): candidates.append(candidate)
                if pos  <= EARLY_CANDIDATE_CUTOFF: early.add(candidate)
                if pos >= num_sentences - LATE_CANDIDATE_CUTOFF: late.add(candidate)
    return prune(candidates, num_sentences), early, late

def add_alternative(pattern, alternative):
    return r'(?:'+pattern+')|(?:'+alternative+')'

TR_PAT = r'(?:last year|last week|sunday|monday|tuesday|wednesday|thursday|friday|saturday)'
TR_PAT = add_alternative(TR_PAT, r'(tomorrow|yesterday|sunday|monday|tuesday|wednesday|thursday|friday|saturday|last|next) (morning|afternoon|evening|night|noon|day|month|week|year|(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?))')
TR_RE = re.compile(TR_PAT, re.IGNORECASE)
def remove_time_ref(candidate):
    '''
    Returns a version of the given candidate phrase with inappropriate
    time references removed.
        @param candidate: The candidate phrase.    
    '''
    return TR_RE.sub('',candidate).strip()
# 0.35 inches Colors || 0.35 pounds Dimensions || 16 GB || 512 MB 
QT_PAT = r'[+-]?((\d+(\.\d*)?)|\.\d+)([eE][+-]?[0-9]+)? ?K|inches|pounds|meters|miles|yards|gb|mb|kb|Dimensions|Storage|pitches'
#QT_PAT = r'[+-]?((\d+(\.\d*)?)|\.\d+)([eE][+-]?[0-9]+)?'
QT_RE = re.compile(QT_PAT, re.IGNORECASE)
def remove_quantity_ref(candidate):
    '''
    Returns a version of the given candidate phrase with inappropriate
    quantity references removed.
        @param candidate: The candidate phrase.    
    '''
    return QT_RE.sub('',candidate).strip()

def contract_pos(tagged_sentence):
    '''
    Returns a version of the given POS tagged sentence with split
    possessives combined.
        @param tagged_sentence: The POS tagged sentence.    
    '''
    ts = []
    for token, tag in tagged_sentence:
        if token == "'s":  
            try:
                prev_token, prev_tag = ts.pop(-1)   
                prev_token += "'s"
                ts.append((prev_token, prev_tag))
            except: ts.append((token, tag))
        else: ts.append((token, tag))
    return ts 
# ---------------------------------------------------------------------------- #
# Methods to filter candidate collections and tokens within a candidate.       #
# ---------------------------------------------------------------------------- #
def prune(candidates, number_of_sentences):
    '''
    Returns a filtered version of the given collection of candidates, removing
    infrequent single word candidates from large documents and replacing 
    candidates with many words by shortened versions.
        @param candidates: The candidate key concepts.    
        @param number_of_sentences: The number of sentences in the underlying
            document.    
    '''
    fdist = FreqDist(candidates)
    results = []
    for candidate in candidates:
        tokens = candidate.split(' ')
        trim(tokens)
        if tokens:
            num_tokens = len(tokens)
            count = fdist[candidate]
            if num_tokens == 1:
                if len(candidate) == 1: continue
                if number_of_sentences >= SINGLE_WORD_FREQ_CUT_OFF and count == 1: continue
            if 'of' in tokens and num_tokens > 5: candidate = ' '.join(tokens[-5:])
            elif num_tokens > 4: candidate = ' '.join(tokens[-4:])
            else:  candidate = ' '.join(tokens)
            #candidate = ' '.join(tokens)
            results.append(candidate)
    return set(results)
#INITIAL_STOP_WORD_PATTERN = r"(?:\d+\-inch(es)?)|created|use|such|on|attended|unlocked|seen|cheaper|fastest|best|and|been|cent|one|two|three|four|or|but|new|nor|other|own|large|late|largest|less|least|many|more|most|inc\.|(?:'s$)"
INITIAL_STOP_WORD_PATTERN = r"a|(?:\$?[\d\.]+)|created|use|such|on|attended|unlocked|seen|cheaper|fastest|best|and|been|cent|one|two|three|four|or|but|new|nor|other|own|large|late|largest|less|least|many|more|most|inc\.|(?:'s$)"
INITIAL_STOP_WORD_RE = re.compile(INITIAL_STOP_WORD_PATTERN, re.IGNORECASE)
#TERMINAL_STOP_WORD_PATTERN = r"ing$|w/out|(?:^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$)|(?:'s$)|today|(?:\d+$)|reports|tomorrow"
TERMINAL_STOP_WORD_PATTERN = r"ing$" \
    + "|w/out" \
    + "|today|tomorrow|yesterday|sunday|monday|tuesday|wednesday|thursday|friday|saturday" \
    + "|reports|roll|north|vs\.?"
TERMINAL_STOP_WORD_RE = re.compile(TERMINAL_STOP_WORD_PATTERN, re.IGNORECASE)
def trim(tokens):    
    '''
    Trims initial and final stop words from the list of words.
        @param tokens: The list of words.    
    '''
    while tokens and INITIAL_STOP_WORD_RE.search(tokens[0]):
        tokens.pop(0)   
    while tokens and TERMINAL_STOP_WORD_RE.search(tokens[-1]):
        tokens.pop(-1)   
    if tokens:
        token = tokens.pop(-1)
        token = re.sub(r"'s$",'', token)
        tokens.append(token)
# ---------------------------------------------------------------------------- #
# Methods to validate a candidate key concept.                                 #
# ---------------------------------------------------------------------------- #

BAD_START_STRINGS = ["s "]
def begins_with_bad_string(candidate):
    for string in BAD_START_STRINGS:
        if string in candidate: return True
    return False

BAD_STRINGS = ["'''", '©', '#', '@', ' .', '·', '•', '|', '//', '~', '>>', '%', 'o.', ',', '™', '§']
def contains_bad_string(candidate):
    for string in BAD_STRINGS:
        if string in candidate: return True
    return False

AMOUNT_PAT = r'^(?:\d+|\d+\.\d+|\.\d+) ?(?:billion|bn|million|m|%|mph|seconds?)?$'
AMOUNT_RE = re.compile(AMOUNT_PAT, re.IGNORECASE)
def is_amount(candidate):
    '''
    Returns true if the given candidate key concept is an amount; false, otherwise.
        @param candidate: The candidate key concept.    
    '''
    if AMOUNT_RE.match(candidate): return True
    return False

STOPWORDS = set(['gem', 'http', 'image', 'move', 'price', 'results', 'said', 'see', 'time', 'year'])
def is_stopword(candidate):
    '''
    Returns true if the given candidate key concept is a stopword; false, otherwise.
        @param candidate: The candidate key concept.    
    '''
    if candidate.strip().lower() in STOPWORDS: return True
    return False

# 12.30 p.m. || 3 p.m.
TIME_PAT = r'^(([0-9])|([0-1][0-9])|([2][0-3]))(\.|:)(([0-9])|([0-5][0-9]))( (a|p)\.?m\.?)?$'
TIME_PAT = add_alternative(TIME_PAT, r'^\d+ (a|p)\.?m\.?$')
TIME_PAT = add_alternative(TIME_PAT, r'^\d\d:\d\d(:\d\d)? EDT$')
TIME_RE = re.compile(TIME_PAT, re.IGNORECASE)
def is_time(candidate):
    '''
    Returns true if the given candidate key concept is an amount; false, otherwise.
        @param candidate: The candidate key concept.    
    '''
    if TIME_RE.match(candidate): return True
    return False
# April 06
MONTH_PATTERN = r'(?:Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?|January|February|March|April|May|June|July|August|September|October|November|December)'
DATE_PAT = r'^'+MONTH_PATTERN+r' \d?\d(,? \d\d\d\d)?$'
DATE_PAT = add_alternative(DATE_PAT, r'^\d+ '+MONTH_PATTERN+r'(,? \d\d\d\d)?$')
DATE_PAT = add_alternative(DATE_PAT, r'^(:?next|last|first|coming)(?: (?:few|couple of|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+))? (?:seconds?|minutes?|hours?|days?|weeks?|months?|quarters?|years?|seasons?|(?:\d{4}))$')
DATE_RE = re.compile(DATE_PAT, re.IGNORECASE)
def is_date(candidate):
    '''
    Returns true if the given candidate key concept is an amount; false, otherwise.
        @param candidate: The candidate key concept.    
    '''
    if DATE_RE.match(candidate): return True
    return False

URL_FINDERS = [
    re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?/[-A-Za-z0-9_\\$\\.\\+\\!\\*\\(\\),;:@&=\\?/~\\#\\%]*[^]'\\.}>\\),\\\"]"),
    re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?"),
    re.compile("(~/|/|\\./)([-A-Za-z0-9_\\$\\.\\+\\!\\*\\(\\),;:@&=\\?/~\\#\\%]|\\\\)+"),
    re.compile("'\\<((mailto:)|)[-A-Za-z0-9\\.]+@[-A-Za-z0-9\\.]+"),
]
def contains_url(candidate):
    '''
    Returns true if the given candidate key concept is an amount; false, otherwise.
        @param candidate: The candidate key concept.    
    '''
    for finder in URL_FINDERS:
        if finder.search(candidate): return True
    return False

VALIDITY_CHECKS = [begins_with_bad_string, contains_bad_string, contains_url, is_amount, is_date, is_stopword, is_time]  
VALIDITY_CHECKS = []  
def valid_candidate(candidate):
    '''
    Returns True if the given candidate key concept is valid; otherwise,
    False.
    '''
    for f in VALIDITY_CHECKS:
        if f(candidate): return False
    return True
