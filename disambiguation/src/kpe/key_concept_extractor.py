# -*- coding: utf-8 -*-
'''
Tools for extracting key concepts from a document.
'''
from collections import defaultdict
from pygraph.classes.digraph import digraph
from candidate_extractor import extract_candidates, valid_candidate
import re
import time
from util import tag_document

EARLY_CANDIDATE_WEIGHT = 3.0
LATE_CANDIDATE_WEIGHT = 0.5
MAX_NUM_CONCEPTS = 10
NUM_CONCEPTS_FACTOR = 7
WORD_WIDTH = 3
def extract_kc(doc, headline=None):
    '''
    Extracts key concepts from the given document and adds them to the given
    tag map.
        @param doc: The document from which to extract key concepts.    
        @param headline: The title of the document.    
    '''
    if headline: doc = headline+'.  '+doc
    tokenized_sentences, tagged_sentences = tag_document(doc.lower())
    num_concepts = min(sum([len(x) for x in tokenized_sentences])/NUM_CONCEPTS_FACTOR, MAX_NUM_CONCEPTS)
    candidates, early_candidates, late_candidates = extract_candidates(tagged_sentences)
    candidates = wash_entities(candidates, [])
    candidates = [[word.strip() for word in candidate.split(' ')] for candidate in candidates]
    keywords = set(sum(candidates, []))
    wg = WordGraph(tokenized_sentences, keywords, WORD_WIDTH)
    ranker = wg.rank_words()
    scores = {}
    for words in candidates:
        key_concept = ' '.join(words)
        scores[key_concept] = 0.0
        for token in words: scores[key_concept] += ranker[token]
        if ' '.join(words) in early_candidates: scores[key_concept] *= EARLY_CANDIDATE_WEIGHT
        elif ' '.join(words) in late_candidates: scores[key_concept] *= LATE_CANDIDATE_WEIGHT
    phrases = [phrase for phrase in scores.keys() if valid_candidate(phrase)]
    ranked = sorted(phrases, key=lambda record: scores[record], reverse=True)
    return set(ranked[:num_concepts])

SPECIAL_CHARACTERS_PAT = '(\.|\^|\$|\*|\+|\?|\}|\{|\\|\[|\]|\||\(|\)|\:)'
SPECIAL_CHARACTERS_RE = re.compile(SPECIAL_CHARACTERS_PAT)
def wash_entities(candidates, blacklist):
    candidates = candidates.difference(blacklist)
    components = []
    for entity in blacklist:
        escaped_entity = SPECIAL_CHARACTERS_RE.sub(r'\\\1', entity)
        components.append(r'(?:^%s)|(?:%s$)' % (escaped_entity, escaped_entity))
    pattern = '|'.join(components)
    ne_reg_exp = re.compile(pattern)
    pos_reg_exp = re.compile(r"(?:^'s)|(?: 's)")
    results = set([])
    for candidate in candidates:
        previous = None
        current = candidate
        while not current == previous:
            previous = current
            current = ne_reg_exp.sub('',current).strip()
            current = pos_reg_exp.sub('',current).strip()
        if current: results.add(current)
    return results
    
class WordGraph(object):
    """
    Encapsulation of a directed graph with weighted edges built from 
    a document according to word co-occurrences. Links are created by
    sliding a window across the document, adding links at each 
    position from the first word pointing to other words within the window.
    """

    def __init__(self, sentences, keywords, window_width):
        """
        Initializes this word graph.        
            @param sentences: The sentences used to determine word co-occurance.
            @param keywords: The keywords defining the nodes of this word graph.
            @param window_width: The size of the window used to determine word co-occurance.
        """
        self.sentences = sentences
        self.window_width = window_width
        self.graph = digraph()
        self.graph.add_nodes(keywords)
        self._add_edges(keywords)
        
    def _add_edges(self, keywords):
        """
        Adds edges to this word graph by sliding a window across each of the given
        sentences and linking the first word in the window to other words within 
        the window.
            @param keywords: The keywords defining the nodes of this word graph.
        """
        weight_dict = defaultdict(int)
        for sentence in self.sentences:
            i = 0
            while i < len(sentence) - 1:
                
                u = sentence[i]
                i += 1
                if not u in keywords: continue
                for v in sentence[i:i+self.window_width-1]:
                    if v in keywords:
                        weight_dict["%s::%s" % (u,v)] += 1
        for key, value in weight_dict.items():
            if value > 0:
                self.graph.add_edge(key.split('::'), wt=value)
        self.out_wt = defaultdict(int)
        for node in self.graph.nodes():
            if self.graph.node_order(node) > 0:
                for nbd in self.graph.neighbors(node):
                    self.out_wt[node] += self.graph.edge_weight((node, nbd))
    
    def rank_words(self, damping_factor=0.85, max_iterations=100, min_delta=0.001):
        """
        Builds and returns a dictionary mapping words to scores.  Scores are computed
        by applying an adaptation of Google's pagerank algorithm to this word graph.
            @param damping_factor: The damping factor applied in each iteration.
            @param max_iterations: The maximum number of iterations.
            @param min_delta: The smallest variation required to have a new iteration..
        """
        nodes = self.graph.nodes()
        graph_size = len(nodes)
        if graph_size == 0: return {}
        ranker = dict.fromkeys(nodes, 1.0/graph_size)        
        bias = 1.0-damping_factor
        i = 0
        while i < max_iterations:
            i += 1
            diff = 0
            for node in nodes:
                rank = 0.0
                for neighbor in self.graph.incidents(node):
                    rank += ranker[neighbor] *  self.graph.edge_weight((neighbor, node)) / self.out_wt[neighbor]
                rank *= damping_factor
                rank += bias
                diff += abs(ranker[node] - rank)
                ranker[node] = rank
            if diff < min_delta:
                break
        return ranker

import codecs
from os.path import join
from os import listdir

path = "/opt/disambiguation/data/jr-docs"
filenames = [join(path, d) for d in sorted(listdir(path))]
for file in filenames:
  try:
    text = data = codecs.open(file, 'r', 'Windows-1252').read()
  except:
    text = data = codecs.open(file, 'r', 'UTF-8').read()    
  print extract_kc(text)