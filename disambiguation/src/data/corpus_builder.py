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
    
  def get_context(self, surface_form, window_radius=50):
    """Returns the list of the tokens in this document appearing in a window 
    with the given width and centered at the first occurrence of the given surface form,
    if the form appears in this document; otherwise, the empty list."""
    words = word_tokenize(surface_form.lower())
    nwords = len(words)
    tokens = self.get_tokens()
    position = 0
    while position + nwords <= len(tokens):
      if tokens[position:position + nwords] == words:
        return tokens[position - window_radius : position + nwords  + window_radius]
      position += 1
    return []
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "%s -->> %s" %(self.doc_id, self.get_text())

class LabeledDocument(Document):
  """Generic interface to a text document with an id and a label."""        
  def __init__(self, doc_id, label):
    super(LabeledDocument, self).__init__(doc_id)
    self.label = label
  
  def __unicode__(self):
    return "%s [%s] -->> %s" %(self.doc_id, self.label, self.get_text())

class TextBackedDocument(Document):
  """Simple text backed document."""        

  def __init__(self, doc_id, doc_text):
    super(TextBackedDocument, self).__init__(doc_id)
    self.doc_text = doc_text
    
  def get_text(self):
    """Returns the text of this document."""
    return self.doc_text

class LabeledTextBackedDocument(LabeledDocument):
  """Simple text backed labeled document."""        

  def __init__(self, doc_id, label, doc_text):
    super(LabeledTextBackedDocument, self).__init__(doc_id, label)
    self.doc_text = doc_text
    
  def get_text(self):
    """Returns the text of this document."""
    return self.doc_text

class LabeledListBackedDocument(LabeledDocument):
  """Simple text backed labeled document."""        

  def __init__(self, doc_id, label, tokens):
    super(LabeledListBackedDocument, self).__init__(doc_id, label)    
    self.doc_text = None
    self.tokens = tokens
    
  def get_text(self):
    """Returns the text of this document."""
    if not self.doc_text: self.doc_text = " ".join(self.tokens)
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

class FileBackedLabeledDocument(LabeledDocument):
  """Simple file backed document."""        

  def __init__(self, doc_id, doc_path, label, store_content=False):
    super(FileBackedLabeledDocument, self).__init__(doc_id, label)
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


class CollectionBackedCorpus(Corpus):
  """Labeled document collection backed BOW corpus."""

  def __init__(self, documents, description, no_below=4, no_above=0.5, keep_n=2500, store_content=False):
    super(CollectionBackedCorpus, self).__init__(description, no_below, no_above, keep_n)
    self.documents = documents

class LabeledCorpus(DirectoryBackedCorpus):
  """Directory backed labeled BOW corpus."""

  def __init__(self, path, description, no_below=4, no_above=0.5, keep_n=2500, store_content=False):
    super(LabeledCorpus, self).__init__(path, description, no_below, no_above, keep_n)
    self.path = path
    self.documents = []
    for d in sorted(listdir(path)):
	directory = join(path, d)
	for f in sorted(listdir(directory)):
      		self.documents.append(FileBackedLabeledDocument(f, join(directory , f), d, store_content))
  
def build_discrimination_corpus(path, form):
  corpus = LabeledCorpus(path, "%s corpus" % form, store_content=True)	
  texts = [document.get_context(form) for document in corpus.documents]
  documents = []
  position = 0
  for text in texts:
    document = corpus.documents[position]
    documents.append(LabeledListBackedDocument(document.doc_id, document.label, text))
    position += 1
  return CollectionBackedCorpus(documents, "%s context corpus" % form)
	
