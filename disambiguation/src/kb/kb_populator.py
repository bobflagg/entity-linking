'''
1. update your Python path:
  export PYTHONPATH=/home/disambiguation/entity-linking/disambiguation/src
2. move to the directory containing the collector script:
  cd /home/disambiguation/entity-linking/disambiguation/src/kb
3. run the script:
  python kb_populator.py <path to config file> [options]
for example:
  python kb_populator.py /home/disambiguation/config/kb-population.cfg -t
Available options are:
  
    -k          Update keyphrase features
    -s          Update surface form data
    -t          Update document topic model
    -v          Update context vector space model

The configuration file is pretty simple and the only thing that you'll probably want
to change is the search-pattern entry in the main section, which specifies the directories
below
	/home/disambiguation/corpus/entity
to process.  Currently it is set with
	search-pattern: js-*
so all the John Smith directories are processed.

'''
import codecs
import ConfigParser
from data.corpus_builder import DirectoryBackedCorpus
from data.domain import SUBTYPES
from gensim import corpora, models
import glob
import json as simplejson
from kpe.key_concept_extractor import extract_kc
from nltk.tokenize import sent_tokenize, word_tokenize
import optparse
import os
import time

def build_document_topic_model(home, config, update=False):
  '''
  Builds and stores an LDA topic model for the ambient document corpus.
  '''
  path = '%s/output/ambient-document.lda' % home
  if update or not os.path.exists(path):
    print "Building and storing ambient document topic model."
    start = time.time()
    corpus_directory = config.get('document-topic-model','directory')
    corpus_path = "%s/corpus/%s" % (home, corpus_directory)
    description = "Ambient document corpus"
    no_below = config.getint('document-topic-model','no-below')
    no_above = config.getfloat('document-topic-model','no-above')
    corpus = DirectoryBackedCorpus(corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('%s/output/ambient-document.dict' % home)
    corpora.MmCorpus.serialize('%s/output/ambient-document.mm' % home, corpus)
    num_topics = config.getint('document-topic-model','num-topics')
    lda = models.ldamodel.LdaModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
    lda.save(path)
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)
  else:
    print "Ambient document topic model appears to be up-to-date."

def build_context_vector_space_model(home, config, update=False):
  '''
  Builds and stores a TF-IDF vector space model for the ambient context corpus.
  '''
  path = '%s/output/ambient-context-vector-space-model.mm' % home
  if update or not os.path.exists(path):
    print "Building and storing ambient context vector space model."
    start = time.time()
    corpus_directory = config.get('context-vector-space-model','directory')
    corpus_path = "%s/corpus/%s" % (home, corpus_directory)
    description = "Ambient context corpus"
    no_below = config.getint('context-vector-space-model','no-below')
    no_above = config.getfloat('context-vector-space-model','no-above')
    corpus = DirectoryBackedCorpus(corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('%s/output/ambient-context.dict' % home)
    corpora.MmCorpus.serialize('%s/output/ambient-context.mm' % home, corpus)
    tfidf = models.TfidfModel(corpus)
    tfidf.save(path)
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)
  else:
    print "Ambient context vector space model appears to be up-to-date."

def collect_entity_surface_form_data(directory, document_count_map, mention_count_map):
  '''
  Collects and stores all surface forms appearing in feature entity kb files.
  '''
  path = "%s/features-entities.json" % directory
  fp = codecs.open(path, 'r', 'UTF-8')
  data = simplejson.load(fp)
  for item in data:
    subtype = item['subtype']
    if subtype in SUBTYPES: print item
  fp.close()

def collect_surface_form_data(home, config, update=False):
  '''
  Collects and stores surface form for all specified (with a search pattern) entities.
  '''
  path = '%s/output/surface-forms.txt' % home
  if update or not os.path.exists(path):
    print "Collecting and storing KB surface forms."
    start = time.time()
    document_count_map = {}
    mention_count_map = {}
    # process all entities
    search_pattern = "%s/output/entity/%s" % (home, config.get('surface-form','search-pattern'))
    for directory in glob.glob(search_pattern):
      collect_entity_surface_form_data(directory, document_count_map, mention_count_map)  
      break  
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)
  else:
    print "KB surface list appears to be up-to-date."

def extract_topic_features(output_directory, documents , dictionary, model, tolerance):
  '''
  Extracts and stores topic features from the given documents, using the
  provided dictionary and topic model.
  '''
  print "\t- topic features."
  corpus = SimpleCorpus(documents , dictionary)
  ndocs = len(documents)
  feature_data = {'number-of-documents':ndocs, 'topics':{}}
  map = {}
  for document in model[corpus]:
    for topic, weight in document: 
      if weight > tolerance:
        if not topic in map: map[topic] = []
        map[topic].append(weight)
  for key in map.keys(): map[key] = sorted(map[key], key=lambda x: -x)
  data = {'number-of-documents':ndocs, 'topics':map}
  fp = codecs.open("%s/features-topics.json" % output_directory, 'w', 'UTF-8')
  simplejson.dump(data, fp, indent=4)
  fp.close()

from data.domain import FeatureSet
import re
def extract_entity_features(output_directory, f='Proximity_AllCollapsedContext_JS', surface_forms = ['Smith', '^John$']):
  '''
  Extracts and stores entity features from the given file.
  '''
  print "\t- entity features."
  surface_form_pattern = "|".join(surface_forms)
  surface_form_re = re.compile(surface_form_pattern, re.IGNORECASE)
  mentions_path = '%s/%s' % (output_directory, f)
  mentions_data = codecs.open(mentions_path, 'r', 'UTF-8').readlines()
  features = []
  finished_doc = True
  for line in mentions_data: 
    line = line.strip()
    if line:
      if finished_doc: 
        #doc_id, path = line.split(":")
        #doc_id = doc_id.strip()
        #path = path.strip()[1:-1]
        #doc_name = path.split('/')[-1]
        finished_doc = False
      else:
        data = line.split('\t')
        sf = data[0].split(':')[0]
        if surface_form_re.search(sf): 
          for item in data[1:-1]:
            fs = FeatureSet(item)
            if not surface_form_re.search(fs.entity.phrase): features.append(fs.to_dict())
    else: 
      finished_doc = True
  fp = codecs.open("%s/features-entities.json" % output_directory, 'w', 'UTF-8')
  simplejson.dump(features, fp, indent=4)
  fp.close()

def extract_keyphrase_features(output_directory, documents, max_keyphrases, update=False):
  '''
  Extracts and stores keyphrase features from the given documents.
  '''
  path = "%s/features-keyphrases.json" % output_directory
  if update or not os.path.exists(path):
    print "\t- updating keyphrase features."
    ndocs = len(documents)
    map = {}
    for document in documents:
      keyphrases = extract_kc(document)
      for keyphrase in keyphrases:
        if not keyphrase in map: map[keyphrase] = 0
        map[keyphrase] += 1
    data = {'number-of-documents':ndocs, 'keyphrases':map}
    fp = codecs.open(path, 'w', 'UTF-8')
    simplejson.dump(data, fp, indent=4)
    fp.close()
  else:
    print "\t- keyphrase features appear to be up-to-data."

class SimpleCorpus(object):
  def __init__(self, docs, dictionary):
    self.docs = docs
    self.dictionary = dictionary

  def __iter__(self):
    for doc in self.docs:
      sentences = sent_tokenize(doc.lower())
      tokenized_sentences = [word_tokenize(x) for x in sentences] 
      tokens = sum(tokenized_sentences, [])
      yield self.dictionary.doc2bow(tokens)

def extract_entity_kb_data(home, config, opts, doc_dictionary, doc_model, entity):
  print " ->> Extracting features for entity %s." % entity
  kb_entry = {}
  output_directory = "%s/output" % home
  target_directory = "%s/entity/%s" % (output_directory, entity)
  if not os.path.exists(target_directory): os.makedirs(target_directory)
  # get entity documents
  path = "%s/corpus/entity/%s" % (home, entity)
  documents = []
  for d in [f for f in os.listdir(path) if '.' in f]:
    documents.append(codecs.open("%s/%s" % (path, d), 'r', 'UTF-8').read())
  # topic features
  tolerance = config.getfloat('document-topic-model','feature-tolerance')
  extract_topic_features(target_directory, documents , doc_dictionary, doc_model, tolerance)
  # keyphrase features
  max_keyphrases= config.getint('keyphrase','max-keyphrases')
  extract_keyphrase_features(target_directory, documents, max_keyphrases, opts.update_keyphrase_features)
  # entity features
  #extract_entity_features(target_directory, f='Proximity_AllCollapsedContext_JS', surface_forms = ['Smith', '^John$'])
  extract_entity_features(target_directory)

def extract_kb_data(home, config, opts):
  print "Extracting features."
  output_directory = "%s/output" % home
  doc_dictionary = corpora.Dictionary.load('%s/ambient-document.dict' % output_directory)
  doc_model = models.ldamodel.LdaModel.load('%s/ambient-document.lda' % output_directory)
  # process all entities
  search_pattern = "%s/corpus/entity/%s" % (home, config.get('main','search-pattern'))
  for d in glob.glob(search_pattern):
    extract_entity_kb_data(home, config, opts, doc_dictionary, doc_model, d.split('/')[-1])

if __name__ == "__main__":
  # Parse command-line options.
  parser = optparse.OptionParser()

  parser.add_option("-b", help="Skip extracting KB data", dest='skip_extracting_kb_data', default=False, action='store_true')
  parser.add_option("-k", help="Update keyphrase features", dest='update_keyphrase_features', default=False, action='store_true')
  parser.add_option("-s", help="Update surface form data", dest='update_surface_form_data', default=False, action='store_true')
  parser.add_option("-t", help="Update document topic model", dest='update_topic_model', default=False, action='store_true')
  parser.add_option("-v", help="Update context vector space model", dest='update_vector_model', default=False, action='store_true')
  (opts, args) = parser.parse_args()
  # Read configuration info:
  if len(args) == 1: config_path = args[0]
  else: config_path = "/home/disambiguation/config/kb-population.cfg"
  config = ConfigParser.ConfigParser()
  config.read(config_path) 
  home = config.get('main','home')
  # Build ambient document corpus topic model
  build_document_topic_model(home, config, update=opts.update_topic_model)
  # Build ambient context corpus vector space model
  build_context_vector_space_model(home, config, update=opts.update_vector_model)
  # Extract features
  if not opts.skip_extracting_kb_data: extract_kb_data(home, config, opts)
  # Collect surface forms
  collect_surface_form_data(home, config, update=opts.update_surface_form_data)