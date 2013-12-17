import codecs
from gensim import corpora
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from os.path import join
from os import listdir

class Document(object):
  """Generic interface to a text document with an id."""        
  def __init__(self, doc_id):
    self.doc_id = doc_id
    self.tokens = None
    
  def get_id(self):
    """Returns the id of this document."""
    return self.doc_id
    
  def get_text(self):
    """Returns the text of this document."""
    raise Exception("Method not implemented!")
    
  def get_tokens(self):
    """Returns a list of the tokens of this document."""
    if not self.tokens:
      sentences = sent_tokenize(self.get_text().lower())
      tokenized_sentences = [word_tokenize(x) for x in sentences] 
      self.tokens = sum(tokenized_sentences, [])
    return self.tokens
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "%s -->> %s" %(self.doc_id, self.get_text())

class TextBackedDocument(Document):
  """Simple text backed document."""        

  def __init__(self, doc_id, doc_text):
    super(TextBackedDocument, self).__init__(doc_id)
    self.doc_text = doc_text
    
  def get_text(self):
    """Returns the text of this document."""
    return self.doc_text

class FileBackedDocument(Document):
  """Simple file backed document."""        

  def __init__(self, doc_id, doc_path, store_content=False):
    super(FileBackedDocument, self).__init__(doc_id)
    self.doc_path = doc_path
    self.store_content = store_content
    if self.store_content: self.text = codecs.open(self.doc_path, 'r', 'UTF-8').read()
    
  def get_text(self):
    """Returns the text of this document."""
    if self.store_content: return self.text
    return codecs.open(self.doc_path, 'r', 'UTF-8').read()

class Corpus(object):
  """Generic BOW corpus."""

  def __init__(self, description, no_below=4, no_above=0.5, keep_n=2500):
    self.description = description
    self.no_below = no_below
    self.no_above = no_above
    self.keep_n = keep_n
    self.dictionary = None
    self.documents = []
        
  def initialize_dictionary(self):
    texts = [[token for token in document.get_tokens() if len(token) > 1] for document in self.documents]
    dictionary = corpora.Dictionary(texts)
    stopword_ids = [dictionary.token2id[word] for word in stopwords.words('english') if word in dictionary.token2id]
    dictionary.filter_tokens(bad_ids=stopword_ids)
    dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)
    dictionary.compactify()
    self.dictionary = dictionary
        
  def __iter__(self):
    if not self.dictionary: self.initialize_dictionary()
    for document in self.documents:
      yield self.dictionary.doc2bow(document.get_tokens())
    
  def __str__(self):
    documents = [str(document) for document in self.documents[:11]]
    documents.append("...")
    #documents.extend([str(document) for document in self.documents[-2:]])
    return "%s:\n\t%s" % (self.description, "\n\t".join(documents))
  
  def __len__(self): return len(self.documents)
  
class FileBackedCorpus(Corpus):
  """File backed BOW corpus, with one document per line."""

  def __init__(self, path, description, no_below=4, no_above=0.5, keep_n=2500):
    super(FileBackedCorpus, self).__init__(description, no_below, no_above, keep_n)
    self.path = path
    data = codecs.open(path, 'r', 'UTF-8')
    self.documents = []
    for line in data: 
      line = line.strip()
      if line:
        doc_id, doc_text = line.split("\t")
        self.documents.append(TextBackedDocument(doc_id, doc_text))

class DirectoryBackedCorpus(Corpus):
  """Directory backed BOW corpus."""

  def __init__(self, path, description, no_below=4, no_above=0.5, keep_n=2500, store_content=False):
    super(DirectoryBackedCorpus, self).__init__(description, no_below, no_above, keep_n)
    self.path = path
    self.documents = []
    for d in sorted(listdir(path)):
      self.documents.append(FileBackedDocument(d, join(path, d), store_content))
