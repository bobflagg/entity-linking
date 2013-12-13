import codecs
import numpy as np
from os.path import join
from os import listdir
from sklearn.utils import check_random_state

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self
        
def build_corpus(path, description=None, load_content=True, shuffle=True, 
  encoding='utf-8', decode_error='strict', random_state=0):
  """Load text files from a directory with the given path.
  
  This function does not try to extract features into a numpy array or
  scipy sparse matrix. In addition, if load_content is false it
  does not try to load the files in memory.
  
  If you set load_content=True, you should also specify the encoding of
  the text using the 'encoding' parameter. For many modern text files,
  'utf-8' will be the correct encoding. If you leave encoding equal to None,
  then the content will be made of bytes instead of Unicode, and you will
  not be able to use most functions in `sklearn.feature_extraction.text`.
  
  Parameters
  ----------
  path : string or unicode
      Path to the folder holding the data
  
  description: string or unicode, optional (default=None)
      A paragraph describing the characteristics of the dataset: its source,
      reference, etc.
  
  load_content : boolean, optional (default=True)
      Whether to load or not the content of the different files. If
      true a 'data' attribute containing the text information is present
      in the data structure returned. If not, a filenames attribute
      gives the path to the files.
  
  encoding : string or None (default is 'utf-8')
      If None, do not try to decode the content of the files (e.g. for
      images or other non-text content).
      If not None, encoding to use to decode text files to Unicode if
      load_content is True.
  
  decode_error: {'strict', 'ignore', 'replace'}, optional
      Instruction on what to do if a byte sequence is given to analyze that
      contains characters not of the given `encoding`. Passed as keyword
      argument 'errors' to bytes.decode.
  
  shuffle : bool, optional (default=True)
      Whether or not to shuffle the data: might be important for models that
      make the assumption that the samples are independent and identically
      distributed (i.i.d.), such as stochastic gradient descent.
  
  random_state : int, RandomState instance or None, optional (default=0)
      If int, random_state is the seed used by the random number generator;
      If RandomState instance, random_state is the random number generator;
      If None, the random number generator is the RandomState instance used
      by `np.random`.
  
  Returns
  -------
  data : Bunch
      Dictionary-like object, the interesting attributes are: either
      data, the raw text data, or 'filenames', the files
      holding it, and 'DESCR', the full description of the dataset.
  """
  filenames = [join(path, d) for d in sorted(listdir(path))]
  # convert to array for fancy indexing
  filenames = np.array(filenames)  
  if shuffle:
    random_state = check_random_state(random_state)
    indices = np.arange(filenames.shape[0])
    random_state.shuffle(indices)
    filenames = filenames[indices] 
  if load_content:
    ids = []
    documents = []
    for filename in filenames:
      document_id = filename.strip().split("/")[-1]
      document = codecs.open(filename, 'r', 'UTF-8').read()
      ids.append(document_id.strip())
      documents.append(document.strip())
      return Bunch(filenames=filenames, ids=ids, documents=documents, description=description)
  return Bunch(filenames=filenames, description=description)

        
def build_context_corpus(path, description=None, encoding='utf-8'):
  """Load context data from a file with the given path. Each line of
  the file contains a doc id and text separated by a tab.
  
  Parameters
  ----------
  path : string or unicode
      Path to the file holding the data
  
  description: string or unicode, optional (default=None)
      A paragraph describing the characteristics of the dataset: its source,
      reference, etc.
        
  encoding : string or None (default is 'utf-8')
      If None, do not try to decode the content of the files (e.g. for
      images or other non-text content).
      If not None, encoding to use to decode text files to Unicode if
      load_content is True.
  
  Returns
  -------
  data : Bunch
      Dictionary-like object with the following attributes:
        ids: an array of the document ids;
        documents: and array of the text documents;
        description: a description of the dataset.
  """
  data = codecs.open(path, 'r', 'UTF-8')
  ids = []
  documents = []
  for line in data: 
    line = line.strip()
    if line:
      document_id, document = line.split("\t")
      ids.append(document_id.strip())
      documents.append(document.strip())
  return Bunch(ids=ids, documents=documents, description=description)
