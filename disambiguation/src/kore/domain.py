from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk import regexp_tokenize
import re
import string

STOPWORDS = set(stopwords.words('english'))

class Phrase(object):      
  """
  An indexed phrase.
  """
  def __init__(self, index, form):
    """
    Initialize this keyphrase.
    """
    self.index = index
    self.form = form
    tokens = regexp_tokenize(form, r'\s+', gaps=True)
    tokens = [re.sub(r'[^\w]','',token) for token in tokens if not token in STOPWORDS]
    self.tokens = [token for token in tokens if len(token) > 1]
    
  def __str__(self):
    return self.form.encode('utf-8')
  
  def __unicode__(self):
    return self.form
  
class Keyphrase(object):      
  """
  A keyphrase occurrence in a mention context.
  """
  def __init__(self, mention, phrase, proximity, frequency):
    """
    Initialize this keyphrase.
    """
    self.mention = mention
    self.phrase = phrase
    self.proximity = proximity
    self.frequency = frequency   
    
  def __str__(self):
    return str(self.phrase)
  
  def __unicode__(self):
    return self.phrase
  
class Mention(object):      
  """
  An entity mention in a document.
  """
  
  def __init__(self, mention_id, document, occurrence, index, phrase):
    """
    Initialize this keyphrase.
    """
    self.document = document
    self.occurrence = occurrence
    self.id = mention_id
    self.index = index
    self.phrase = phrase
    self.keyphrases = []
    
  def add_keyphrase(self, phrase, proximity, frequency):
    self.keyphrases.append(Keyphrase(self, phrase, proximity, frequency))
    
  def trim_keyphrases(self, max_allowed):
    keyphrases = sorted(self.keyphrases, key = lambda x: -x.proximity)
    self.keyphrases = keyphrases[:max_allowed]
    
  def keyword_doc(self):
    return " ".join(sum([item.phrase.tokens for item in self.keyphrases], []))
    
  def get_phi(self):
    if not hasattr(self, 'phi'):
      self.phi = sum(keyphrase.proximity for keyphrase in self.keyphrases)
    return self.phi
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    keyphrases = [keyphrase.phrase.form for keyphrase in self.keyphrases]
    return "[%d] %s: %s -->> [%s]" % (self.index, self.id, self.phrase, ", ".join(keyphrases))
  
class KeyphraseDataset(object):      
  """
  A data structure for holding key information about the key phrases 
  extracted from a corpus of entity mentions.
  """
  def __init__(self, group):
    """
    Initialize this key phrase data set.
    """
    self.group = group
    self.next_token_index = 0
    self.next_phrase_index = 0
    self.next_mention_index = 0
    self.mention_map = {}
    self.phrase_map = {}
    self.token_map = {}

  def get_mention(self, document, occurrence):
    mention_id = get_mention_id(document, occurrence)
    if mention_id in self.mention_map: mention = self.mention_map[mention_id]
    else: 
      mention = Mention(mention_id, document, occurrence, self.next_mention_index, "John Smith")
      self.mention_map[mention_id] = mention
      self.next_mention_index += 1
    return mention

  def get_phrase(self, form):
    if form in self.phrase_map: phrase = self.phrase_map[form]
    else: 
      phrase = Phrase(self.next_phrase_index, form)
      self.phrase_map[form] = phrase
      for token in phrase.tokens: self.add_token(token)
      self.next_phrase_index += 1
    return phrase

  def add_keyphrase(self, mention, form, proximity, frequency):
    phrase = self.get_phrase(form)
    mention.add_keyphrase(phrase, proximity, frequency) 

  def add_token(self, token):
    if not (token in self.token_map):
      self.token_map[token] = self.next_token_index
      self.next_token_index += 1

  def mentions(self):
    if not hasattr(self, 'mention_list'):
      self.mention_list = sorted(self.mention_map.values(), key = lambda x: x.index)
    return self.mention_list
  
def get_mention_id(document, occurrence):    
  return "%s::%d" % (document, occurrence)
